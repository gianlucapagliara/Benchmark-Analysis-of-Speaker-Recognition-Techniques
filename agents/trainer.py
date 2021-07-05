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
from utils.misc import print_cuda_statistics
from utils.metrics import AverageMeter
from utils.tuneThreshold import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf
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
                # sampler=self.test_sampler
            )

        # Loss
        LossFunction = importlib.import_module(
            'graphs.losses.'+config.loss_function).__getattribute__('LossFunction')
        self.__loss__ = LossFunction(**vars(config)).to(self.device)

        # Model
        Model = importlib.import_module(
            'graphs.models.' + config.model).__getattribute__(config.model)
        self.__model__ = Model(**vars(config)).to(self.device)

        # Checkpoint Loading (if not found start from scratch)
        if(self.config.get('checkpoint_file', "") != ""):
            self.load_checkpoint(self.config.checkpoint_file)

        # Model Loading (if not found start from scratch)
        self.load_parameters(self.config.initial_model)

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
            self.__model__.load_state_dict(checkpoint['state_dict'])
            self.__optimizer__.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(
                self.config.checkpoint_dir))

    def load_parameters(self, path):

        self_state = self.__model__.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("__S__.", "")  # Vox
                name = name.replace("module.", "")  # AutoSpeech

                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'state_dict': self.__model__.state_dict(),
            'optimizer': self.__optimizer__.state_dict(),
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

            scores, labels, trials = self.validate()

            valid_acc = 1
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

        self.current_iteration = 0
        total_iterations = len(self.train_loader)
        start_time = time.time()

        # for x, y in tqdm_batch:
        for x, y in self.train_loader:
            loop_time = time.time()-start_time

            x = x.transpose(1, 0).to(self.device)
            y = torch.LongTensor(y).to(self.device)

            #prepare_time = time.time()-start_time
            prepare_time = time.time()-start_time-loop_time

            self.__model__.zero_grad()

            if self.mixedprec:
                with autocast():
                    cur_loss, curr_top1 = self.__model__.forward_loss(
                        x, y, self.__loss__, **vars(self.config))
                self.scaler.scale(cur_loss).backward()
                self.scaler.step(self.__optimizer__)
                self.scaler.update()
            else:
                cur_loss, curr_top1 = self.__model__.forward_loss(
                    x, y, self.__loss__, **vars(self.config))
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
            process_time = time.time()-start_time-prepare_time-loop_time

            # Logging
            total = process_time+prepare_time+loop_time
            perc_loop = loop_time/total*100
            perc_proc = process_time/total*100
            perc_prep = prepare_time/total*100
            #tqdm_batch.set_description("Epoch {} | Loss {:f} TEER/TAcc {:2.3f}% | Comput. Eff.: {:.2f}% ".format(self.current_epoch+1, epoch_loss.val, epoch_top1.val, efficiency), refresh=True)
            #tqdm_batch.set_description("Epoch {} | Loss {:f} TEER/TAcc {:2.3f}% | Loop: {:.2f}% - Preparation: {:.2f}% - Process: {:.2f}% ".format(self.current_epoch+1, epoch_loss.val, epoch_top1.val, perc_loop, perc_prep, perc_proc), refresh=True)

            sys.stdout.write("\rEpoch-{} ({}/{}) | Loss {:f} TEER/TAcc {:2.3f}% | Time remaining: {} | Loop: {:.2f}% - Preparation: {:.2f}% - Process: {:.2f}%\n"
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

        all_scores = []
        all_labels = []

        eer = 0
        mindcf = 0

        total_iterations = len(self.test_loader)
        current_iteration = 0
        start_time = time.time()

        tqdm_test = tqdm(self.test_loader, total=len(
            self.test_loader), desc="Epoch {}".format(self.current_epoch+1))
        for ref, com, label in tqdm_test:
            current_iteration += 1

            loop_time = time.time()-start_time

            ref = ref.to(self.device)
            com = com.to(self.device)
            label = int(label.data.cpu().numpy()[0])
            with torch.no_grad():
                score = self.__model__.scoring(
                    ref, com, normalize=self.__loss__.test_normalize)

            all_scores.append(score)
            all_labels.append(label)

            # if(current_iteration==1):
            #     print(score)

            if (current_iteration % self.config.print_interval == 0) or (current_iteration == total_iterations):
                result = tuneThresholdfromScore(
                    all_scores, all_labels, [1, 0.1])
                p_target = 0.05
                c_miss = 1
                c_fa = 1
                fnrs, fprs, thresholds = ComputeErrorRates(
                    all_scores, all_labels)
                mindcf, threshold = ComputeMinDcf(
                    fnrs, fprs, thresholds, p_target, c_miss, c_fa)
                eer = result[1]

            process_time = time.time()-start_time-loop_time

            # Logging
            total = process_time+loop_time
            perc_proc = process_time/total*100

            tqdm_test.set_description("Epoch-{} Validation | VEER {:2.4f}% MDC {:2.5f} | Efficiency {:2.2f}% "
                                        .format(self.current_epoch+1, eer, mindcf, perc_proc))

            start_time = time.time()

        validation_time = tqdm_test.format_dict['elapsed']
        validation_rate = total_iterations / validation_time

        tqdm_test.close()

        self.logger.info("Validation at epoch-{} completed in {:2.1f}s ({:2.1f}samples/s). VEER {:2.4f}% MDC {:2.5f} ".format(
            self.current_epoch+1, validation_time, validation_rate, eer, mindcf))

        return eer

    def finalize(self):
        self.logger.info("Finalizing the operation...")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json(
            "{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
