import sys
import time
import datetime
import importlib
import numpy as np
import GPUtil

import torch
from torch._C import device
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from tensorboardX import SummaryWriter

from agents.base import NNAgent
from utils.misc import print_cuda_statistics
from utils.metrics import *
from datasets.Sampler import Sampler

import tqdm as t
from tqdm import tqdm
import shutil


class Trainer(NNAgent):
    def __init__(self, config):
        super().__init__(config)

        # Dataset
        TrainDataset = importlib.import_module(
            'datasets.' + config.train.dataset).__getattribute__(config.train.dataset)
        self.train_dataset = TrainDataset(device=self.device, **vars(config.train))
        self.sampler = Sampler(
            self.train_dataset, **vars(config.train)) if self.config.train.sampler else None
        self.loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.train.batch_size,
            num_workers=config.train.nDataLoaderThread,
            pin_memory=False,
            drop_last=True,
            sampler=self.sampler,
            # worker_init_fn=worker_init_fn,
        )

        # Loss
        LossFunction = importlib.import_module(
            'graphs.losses.'+config.train.loss_function).__getattribute__('LossFunction')
        self.__loss__ = LossFunction(**vars(config.train)).to(self.device)

        # Optimizer
        Optimizer = importlib.import_module(
            'graphs.optimizers.' + config.train.optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(
            self.__model__.parameters(), **vars(config.train))

        # Scheduler
        Scheduler = importlib.import_module(
            'graphs.schedulers.'+config.train.scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(
            self.__optimizer__, **vars(config.train))
        assert self.lr_step in ['epoch', 'iteration']

    def run(self):
        try:
            if self.print_metrics:
                self.get_metrics()
            self.train()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C... Wait to finalize.")

    def train(self):
        for epoch in range(self.current_epoch, self.config.train.max_epoch):
            self.current_epoch = epoch
            if self.config.train.sampler:
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
        #tqdm_batch = tqdm(self.loader, total=len(self.loader),desc="Epoch {}".format(self.current_epoch+1))

        self.current_iteration = 0
        total_iterations = len(self.loader)
        start_time = time.time()

        # for x, y in tqdm_batch:
        for x, y in self.loader:
            loop_time = time.time()-start_time

            x = x.transpose(1, 0).to(self.device)
            y = torch.LongTensor(y).to(self.device)

            #prepare_time = time.time()-start_time
            prepare_time = time.time()-start_time-loop_time

            self.__model__.zero_grad()

            if self.mixedprec:
                with autocast():
                    cur_loss, curr_top1 = self.__model__.forward_loss(
                        x, y, self.__loss__, **vars(self.config.train))
                self.scaler.scale(cur_loss).backward()
                self.scaler.step(self.__optimizer__)
                self.scaler.update()
            else:
                cur_loss, curr_top1 = self.__model__.forward_loss(
                    x, y, self.__loss__, **vars(self.config.train))
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