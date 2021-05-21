import sys
import time
import datetime
import importlib
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from tensorboardX import SummaryWriter

from agents.base import BaseAgent
from graphs.models.SpeakerNet import SpeakerNet
from utils.misc import print_cuda_statistics
from utils.metrics import AverageMeter
from datasets.Sampler import Sampler

import tqdm as t
from tqdm import tqdm
import shutil


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
        self.config.device = self.device

        # Datasets
        self.to_train = config.train
        if self.to_train:
            TrainDataset = importlib.import_module(
                'datasets.' + config.train_dataset).__getattribute__(config.train_dataset)
            self.train_dataset = TrainDataset(**vars(config))
            self.sampler = Sampler(
                self.train_dataset, **vars(config)) if self.config.sampler else None
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=config.batch_size,
                num_workers=config.nDataLoaderThread,
                sampler=self.sampler,
                pin_memory=False,
                # worker_init_fn=worker_init_fn,
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
        self.__loss__ = self.__loss__.to(self.device)

        # Model
        Model = importlib.import_module(
            'graphs.models.' + config.model).__getattribute__(config.model)
        self.__model__ = SpeakerNet(
            Model(**vars(config)), self.__loss__, self.device, config.nPerSpeaker)
        self.__model__ = self.__model__.to(self.device)

        # Model Loading (if not found start from scratch)
        self.load_checkpoint(self.config.initial_model)

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

        # Counters initialization
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_acc = 0

        # Others
        self.verbose = config.verbose
        self.mixedprec = config.mixedprec

    def load_checkpoint(self, filename):
        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(
                self.config.checkpoint_dir))

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + file_name)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + file_name,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        try:
            if self.to_train:
                self.train()
            if self.to_test:
                self.validate()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C... Wait to finalize.")

    def train(self):
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            if self.config.sampler:
                self.sampler.set_epoch(epoch)

            loss, acc = self.train_one_epoch()

            valid_acc = self.validate()
            is_best = valid_acc > self.best_valid_acc
            if is_best:
                self.best_valid_acc = valid_acc
            self.save_checkpoint(is_best=is_best)

    def train_one_epoch(self):
        # Set the model to be in training mode
        self.__model__.train()

        # Initialize your average meters
        epoch_loss = AverageMeter()
        epoch_top1 = AverageMeter()  # EER or accuracy

        # Initialize tqdm
        #t.tqdm.monitor_interval = 0
        #tqdm_batch = tqdm(self.train_loader, total=len(self.train_loader),desc="Epoch {}".format(self.current_epoch+1))

        start_time = time.time()

        total_iterations = len(self.train_loader)
        # for x, y in tqdm_batch:
        for x, y in self.train_loader:
            loop_time = time.time()-start_time

            x = x.transpose(1, 0)
            y = torch.LongTensor(y).to(self.device)

            #prepare_time = time.time()-start_time
            prepare_time = time.time()-start_time-loop_time

            self.__model__.zero_grad()

            if self.mixedprec:
                with autocast():
                    cur_loss, curr_top1 = self.__model__(x, y)
                self.scaler.scale(cur_loss).backward()
                self.scaler.step(self.__optimizer__)
                self.scaler.update()
            else:
                cur_loss, curr_top1 = self.__model__(x, y)
                cur_loss.backward()
                self.__optimizer__.step()

            if self.lr_step == 'iteration':
                self.__scheduler__.step()

            # Meters update
            epoch_loss.update(cur_loss.item())
            epoch_top1.update(curr_top1.item(), x.size(0))

            self.current_iteration += 1

            self.summary_writer.add_scalar(
                "epoch/loss", epoch_loss.val, self.current_iteration)
            self.summary_writer.add_scalar(
                "epoch/accuracy", epoch_top1.val, self.current_iteration)
            # Logging
            process_time = time.time()-start_time-prepare_time-loop_time
            total = process_time+prepare_time+loop_time
            perc_loop = loop_time/total*100
            perc_proc = process_time/total*100
            perc_prep = prepare_time/total*100
            #tqdm_batch.set_description("Epoch {} | Loss {:f} TEER/TAcc {:2.3f}% | Comput. Eff.: {:.2f}% ".format(self.current_epoch+1, epoch_loss.val, epoch_top1.val, efficiency), refresh=True)
            #tqdm_batch.set_description("Epoch {} | Loss {:f} TEER/TAcc {:2.3f}% | Loop: {:.2f}% - Preparation: {:.2f}% - Process: {:.2f}% ".format(self.current_epoch+1, epoch_loss.val, epoch_top1.val, perc_loop, perc_prep, perc_proc), refresh=True)

            sys.stdout.write("\rEpoch-{} ({}/{}) | Loss {:f} TEER/TAcc {:2.3f}% | Time remaining: {} | Loop: {:.2f}% - Preparation: {:.2f}% - Process: {:.2f}%"
                             .format(self.current_epoch+1, self.current_iteration, total_iterations, epoch_loss.val, epoch_top1.val,
                                     str(datetime.timedelta(seconds=total *
                                                            (total_iterations-self.current_iteration))),
                                     perc_loop, perc_prep, perc_proc))
            sys.stdout.flush()
            start_time = time.time()

        if self.lr_step == 'epoch':
            self.__scheduler__.step()

        # tqdm_batch.close()
        self.logger.info("Training at epoch-{} completed. | Loss {:f} TEER/TAcc {:2.3f}%  ".format(
            self.current_epoch+1, epoch_loss.val, epoch_top1.val))

        return (epoch_loss.val, epoch_top1.val)

    def validate(self):
        self.__model__.eval()

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        # Extract features for every image
        for idx, data in enumerate(self.test_loader):
            inp1 = data[0][0].to(self.device)
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

            ref_feat = feats[data[1]].to(self.device)
            com_feat = feats[data[2]].to(self.device)

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
        self.logger.info("Finalizing the operation...")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json(
            "{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
