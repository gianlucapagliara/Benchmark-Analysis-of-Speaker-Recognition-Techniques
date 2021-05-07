#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optim, test_interval, max_epoch, lr_decay, **kwargs):

	sche_fn = torch.optim.lr_scheduler.StepLR(
		optim, step_size=test_interval, gamma=lr_decay)

	lr_step = 'epoch'

	return sche_fn, lr_step
