import logging
import importlib

import torch
from torch.cuda.amp import GradScaler

from tensorboardX import SummaryWriter

from utils.misc import print_cuda_statistics
from utils.metrics import *

import shutil


class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")

    def run(self):
        """
        The main operator
        :return:
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        raise NotImplementedError


class NNAgent(BaseAgent):
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
        self.print_metrics = self.config.get('print_metrics', True)

        self.loader = None

        self.required_memory = torch.cuda.memory_stats(self.device)['active.all.current']
        # Model
        Model = importlib.import_module(
            'graphs.models.' + config.model).__getattribute__(config.model)
        self.__model__ = Model(**vars(config)).to(self.device)
        self.required_memory = torch.cuda.memory_stats(self.device)['active.all.current'] - self.required_memory

        # Checkpoint Loading (if not found start from scratch)
        if(self.config.get('checkpoint_file', "") != ""):
            self.load_checkpoint(self.config.checkpoint_file)

        # Model Loading (if not found start from scratch)
        if(self.config.get('initial_model', "") != ""):
            self.load_parameters(self.config.initial_model)

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

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n".format(
                self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(
                self.config.checkpoint_dir))

    def load_parameters(self, path):
        loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu)

        if loaded_state.get('state_dict', "") != "":
            loaded_state = loaded_state['state_dict']

        self.__model__.load_state_dict(loaded_state)

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

    def get_metrics(self):
        batch_sizes = self.config['metrics_batches'] if self.config.get(
            'metrics_batches', '') else [1, 2, 4, 8, 16, 32, 64]
        input_size = next(iter(self.loader))[0].squeeze(0).shape

        self.logger.info('Computing metrics...')
        total_params, trainable_params = compute_parameters(self.__model__)
        complexity = compute_complexity(self.__model__, input_size)
        mean_tfp, std_tfp = compute_inference_time(
            self.__model__, input_size, batch_sizes=batch_sizes)
        memory = compute_memory_usage(
            self.__model__, input_size, model_memory=self.required_memory, batch_sizes=batch_sizes)

        self.logger.info(
            '================================================================')
        self.logger.info(
            str(self.__model__.__class__).split('.')[-1].split("'")[0])
        self.logger.info(
            '================================================================')
        self.logger.info('Total params: ' + str(total_params))
        self.logger.info('Trainable params: ' + str(trainable_params))
        self.logger.info('Non-trainable params: ' +
                         str(total_params - trainable_params))
        self.logger.info(
            '----------------------------------------------------------------')
        self.logger.info(f'Average flops cost: {complexity}')
        self.logger.info(
            '----------------------------------------------------------------')
        self.logger.info('Average inference time for every batch size:')
        for b, m, s in zip(batch_sizes, mean_tfp, std_tfp):
            self.logger.info(f'{b}: {m:.2f} +/- {s:.2f}')
        self.logger.info(
            '----------------------------------------------------------------')
        self.logger.info('Memory usage for every batch size:')
        for b, m in zip(batch_sizes, memory):
            self.logger.info(f'{b}: {m: .4f}')
        self.logger.info(
            '================================================================')

    def finalize(self):
        self.logger.info("Finalizing the operation...")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json(
            "{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
