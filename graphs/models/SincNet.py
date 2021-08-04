import torch
import torch.nn as nn
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

    def load_state_dict(self, state_dict):
        if state_dict.get('CNN_model_par', '') != '':
            self.CNN_net.load_state_dict(state_dict['CNN_model_par'])
        if state_dict.get('DNN1_model_par', '') != '':
            self.DNN1_net.load_state_dict(state_dict['DNN1_model_par'])
        if state_dict.get('DNN2_model_par', '') != '':
            self.DNN2_net.load_state_dict(state_dict['DNN2_model_par'])

    def scoring(self, ref, com, normalize=False):
        # Feature extraction
        ref_feat = self.get_dvect(ref).to(self.device)
        com_feat = self.get_dvect(com).to(self.device)

        # Distance
        score = F.pairwise_distance(ref_feat, com_feat)
        score = score.detach().cpu().numpy()
        score = -1 * np.mean(score)

        return score

    def get_dvect(self, signal):
        signal = signal.squeeze(0)

        # split signals into chunks
        beg_samp = 0
        end_samp = self.wlen

        d_vector_dim = self.DNN1_arch['fc_lay'][-1]

        # when loading with torchaudio
        N_fr = int((signal.shape[0]-self.wlen)/(self.wshift))

        sig_arr = torch.zeros([self.batch_size, self.wlen]).float().to(
            self.device).contiguous()
        dvects = Variable(torch.zeros(
            N_fr, d_vector_dim).float().to(self.device).contiguous())

        count_fr = 0
        count_fr_tot = 0
        while end_samp < signal.shape[0]:
            sig_arr[count_fr, :] = signal[beg_samp:end_samp]
            beg_samp = beg_samp + self.wshift
            end_samp = beg_samp + self.wlen
            count_fr = count_fr + 1
            count_fr_tot = count_fr_tot + 1
            if count_fr == self.batch_size:
                inp = Variable(sig_arr)
                out = self.DNN1_net(self.CNN_net(inp))
                dvects[count_fr_tot-self.batch_size:count_fr_tot, :] = out
                count_fr = 0
                sig_arr = torch.zeros([self.batch_size, self.wlen]).float().to(
                    self.device).contiguous()

        if count_fr > 0:
            inp = Variable(sig_arr[:count_fr])
            out = self.DNN1_net(self.CNN_net(inp))
            dvects[count_fr_tot-count_fr-1:count_fr_tot, :] = out

        # averaging and normalizing all the d-vectors
        d_vect_out = torch.mean(
            dvects/dvects.norm(p=2, dim=1).view(-1, 1), dim=0)

        d_vect_out = d_vect_out.unsqueeze(0)
        return d_vect_out

    def preforward(self, x):
        if x.shape == 1:
            x = x.unsqueeze(0)
        return x

    def forward(self, x):
        x = self.preforward(x)
        d_vects = []
        for i in range(0, x.shape[0]):
            signal = x[i, :]
            d_vects.append(self.get_dvect(signal))
        return d_vects

