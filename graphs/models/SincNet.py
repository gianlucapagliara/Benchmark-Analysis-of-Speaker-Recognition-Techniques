import torch
import torch.nn.functional as F

from graphs.models.base import BaseModel
from graphs.models.SincNetBlocks import *

CNN_ARCH = {
    'cnn_N_filt': [80, 60, 60],
    'cnn_len_filt': [251, 5, 5],
    'cnn_max_pool_len': [3, 3, 3],
    'cnn_use_laynorm_inp': True,
    'cnn_use_batchnorm_inp': False,
    'cnn_use_laynorm': [True, True, True],
    'cnn_use_batchnorm': [False, False, False],
    'cnn_act': ['leaky_relu', 'leaky_relu', 'leaky_relu'],
    'cnn_drop': [0.0, 0.0, 0.0],
}

DNN1_ARCH = {
    'fc_lay': [2048, 2048, 2048],
    'fc_drop': [0.0, 0.0, 0.0],
    'fc_use_laynorm_inp': True,
    'fc_use_batchnorm_inp': False,
    'fc_use_batchnorm': [True, True, True],
    'fc_use_laynorm': [False, False, False],
    'fc_act': ['leaky_relu', 'leaky_relu', 'leaky_relu'],
}

DNN2_ARCH = {
    'input_dim': DNN1_ARCH['fc_lay'][-1],
    'fc_lay': [462],
    'fc_drop': [0.0],
    'fc_use_laynorm_inp': False,
    'fc_use_batchnorm_inp': False,
    'fc_use_batchnorm': [False],
    'fc_use_laynorm': [False],
    'fc_act': ['softmax'],
}


class SincNet(BaseModel):
    def __init__(self, device, fs, cw_len, cw_shift, batch_size=128, **kwargs):
        super(SincNet, self).__init__(device)

        self.fs = fs
        self.wlen = int(self.fs*cw_len/1000.00)
        self.wshift = int(self.fs*cw_shift/1000.00)
        self.batch_size = batch_size

        self.CNN_arch = CNN_ARCH
        self.CNN_arch['input_dim'] = self.wlen
        self.CNN_arch['fs'] = self.fs
        self.CNN_net = CNN(self.CNN_arch)

        self.DNN1_arch = DNN1_ARCH
        self.DNN1_arch['input_dim'] = self.CNN_net.out_dim
        self.DNN1_net = MLP(self.DNN1_arch)

        self.DNN2_arch = DNN2_ARCH
        self.DNN2_net = MLP(self.DNN2_arch)

    def get_dvect(self, signal):
        dvect = torch.zeros(
            signal.shape[0], self.DNN1_arch['fc_lay'][-1]).float().to(self.device)

        # batches
        n_pred = signal.shape[0]//self.batch_size
        for i in range(0, n_pred):
            dvect[self.batch_size*i:self.batch_size*(i+1)] = self.DNN1_net(self.CNN_net(
                signal[self.batch_size*i:self.batch_size*(i+1)]))
            torch.cuda.empty_cache()

        # last batch
        if(signal.shape[0] % self.batch_size > 0):
            dvect[self.batch_size*(n_pred-1):] = self.DNN1_net(self.CNN_net(
                signal[self.batch_size*(n_pred-1):]))
            torch.cuda.empty_cache()

        # averaging and normalizing all the d-vectors
        dvect = torch.mean(
            dvect/dvect.norm(p=2, dim=1).view(-1, 1), dim=0)

        return dvect

    def preforward(self, x):
        x = super(SincNet, self).preforward(x)
        if x.shape == 2:
            x = x.unsqueeze(0)
        return x

    def forward(self, x):
        x = self.preforward(x)
        
        d_vects = torch.zeros(
            x.shape[0], self.DNN1_arch['fc_lay'][-1]).float().to(self.device)
        for i in range(0, x.shape[0]):
            d_vects[i, :] = self.get_dvect(x[i, :])
        
        return d_vects
