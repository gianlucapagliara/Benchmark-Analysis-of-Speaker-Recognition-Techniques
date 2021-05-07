import sys
import time
import importlib
import tqdm
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from tensorboardX import SummaryWriter

from agents.base import BaseAgent
from graphs.models.SpeakerNet import SpeakerNet
from utils.misc import print_cuda_statistics

import torchaudio

torchaudio.set_audio_backend("soundfile")


class Trainer(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        # GPU and CUDA
        # Construct the flag and make sure that cuda is available
        self.cuda = torch.cuda.is_available() & self.config.cuda
        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.manual_seed_all(self.config.seed)
            torch.cuda.set_device(self.config.gpu_device)
            self.logger.info("Operation will be on ***** GPU-CUDA ***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.config.seed)
            self.logger.info("Operation will be on ***** CPU ***** ")
        self.gpu = config.gpu_device


        # Datasets
        self.to_train = config.train
        if self.to_train:
            TrainDataset = importlib.import_module(
                'datasets.' + config.train_dataset).__getattribute__(config.train_dataset)
            self.train_dataset = TrainDataset(**vars(config))
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=config.batch_size,
                num_workers=config.nDataLoaderThread,
                #sampler=train_sampler,
                pin_memory=False,
                #worker_init_fn=worker_init_fn,
                drop_last=True,
            )

        self.to_test = config.test
        if self.to_test:
            TestDataset = importlib.import_module(
                'datasets.' + config.test_dataset).__getattribute__(config.test_dataset)
            self.test_dataset = TestDataset(**vars(config))
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=config.nDataLoaderThread,
                drop_last=False,
            )

        # Loss
        LossFunction = importlib.import_module(
            'graphs.losses.'+config.loss_function).__getattribute__('LossFunction')
        self.__loss__ = LossFunction(**vars(config))

        # Model
        Model = importlib.import_module(
            'graphs.models.' + config.model).__getattribute__(config.model)
        self.__model__ = SpeakerNet(
            Model(**vars(config)), self.__loss__, config.nPerSpeaker)

        # Optimizer
        Optimizer = importlib.import_module(
            'graphs.optimizers.' + config.optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(
            self.__model__.parameters(), **vars(config))

        # Scheduler
        Scheduler = importlib.import_module(
            'graphs.schedulers.'+config.scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(
            self.__optimizer__, **vars(config))
        assert self.lr_step in ['epoch', 'iteration']

        # Scaler
        self.scaler = GradScaler()

        # Tensorboard Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir)

        # Others
        self.verbose = config.verbose
        self.mixedprec = config.mixedprec

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        self_state = self.__model__.module.state_dict()
        path = self.config.checkpoint_dir + file_name
        loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        path = self.config.checkpoint_dir + file_name
        torch.save(self.__model__.module.state_dict(), path)

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            if self.to_train:
                self.train()
            if self.to_test:
                self.validate()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C... Wait to finalize.")

    def train(self):
        """
        Main training loop
        :return:
        """
        # Set the model to be in training mode
        self.__model__.train()

        stepsize = self.train_loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0    # EER or accuracy

        tstart = time.time()

        for data, data_label in self.train_loader:

            data = data.transpose(1, 0)

            self.__model__.zero_grad()

            label = torch.LongTensor(data_label).cuda()

            if self.mixedprec:
                with autocast():
                    nloss, prec1 = self.__model__(data, label)
                self.scaler.scale(nloss).backward()
                self.scaler.step(self.__optimizer__)
                self.scaler.update()
            else:
                nloss, prec1 = self.__model__(data, label)
                nloss.backward()
                self.__optimizer__.step()

            loss += nloss.detach().cpu().item()
            top1 += prec1.detach().cpu().item()
            counter += 1
            index += stepsize

            telapsed = time.time() - tstart
            tstart = time.time()

            if self.verbose:
                sys.stdout.write("\rProcessing ({:d}) ".format(index))
                sys.stdout.write("Loss {:f} TEER/TAcc {:2.3f}% - {:.2f} Hz ".format(
                    loss/counter, top1/counter, stepsize/telapsed))
                sys.stdout.flush()

            if self.lr_step == 'iteration':
                self.__scheduler__.step()

        if self.lr_step == 'epoch':
            self.__scheduler__.step()

        return (loss/counter, top1/counter)

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        # Initialize tqdm
        tqdm_batch = tqdm(self.train_loader, total=len(self.train_loader),
                          desc="Epoch-{}-".format(self.current_epoch))

        pass

    def validate(self):
        """
        Model validation
        :return:
        """
        self.__model__.eval()

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        # Extract features for every image
        for idx, data in enumerate(self.test_loader):
            inp1 = data[0][0].cuda()
            ref_feat = self.__model__(inp1).detach().cpu()
            feats[data[1][0]] = ref_feat
            telapsed = time.time() - tstart

            if idx % self.config.print_interval == 0:
                sys.stdout.write("\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(
                    idx, len(self.test_dataset), idx/telapsed, ref_feat.size()[1]))

        print('')
        all_scores = []
        all_labels = []
        all_trials = []
        tstart = time.time()

        # Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split()

            # Append random label if missing
            if len(data) == 2:
                data = [random.randint(0, 1)] + data

            ref_feat = feats[data[1]].cuda()
            com_feat = feats[data[2]].cuda()

            if self.__model__.module.__L__.test_normalize:
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                com_feat = F.normalize(com_feat, p=2, dim=1)

            dist = F.pairwise_distance(
                ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0, 2)).detach().cpu().numpy()

            score = -1 * np.mean(dist)

            all_scores.append(score)
            all_labels.append(int(data[0]))
            all_trials.append(data[1]+" "+data[2])

            if idx % self.config.print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing {:d} of {:d}: {:.2f} Hz".format(
                    idx, len(lines), idx/telapsed))
                sys.stdout.flush()

        return (all_scores, all_labels, all_trials)

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.logger.info("Finalizing the operation...")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json(
            "{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
