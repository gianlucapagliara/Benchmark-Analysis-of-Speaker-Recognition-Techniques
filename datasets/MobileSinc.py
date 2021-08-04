import os
import random
import numpy as np
import soundfile as sf

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.autograd import Variable


class MobileSinc(Dataset):
    def __init__(self, test_list, test_path, max_files=0, **kwargs):
        self.test_path = test_path
        self.test_list = test_list

        self.data_list = []
        self.labels = []
        self.pairs = []

        # Read files
        with open(test_list) as dataset_file:
            lines = dataset_file.readlines()

        for line in lines:
            data = line.split()

            # Append random label if missing
            if len(data) == 2:
                data = [random.randint(0, 1)] + data

            self.labels.append(data[0])
            self.pairs.append([data[1], data[2]])

        if max_files > 0:
            self.pairs = self.pairs[:max_files]
            self.labels = self.labels[:max_files]

    def create_batches_rnd(self, batch_size, wlen, fact_amp):

        # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
        sig_batch=np.zeros([batch_size,wlen])
        lab_batch=np.zeros(batch_size)
        
        rand_amp_arr = np.random.uniform(1.0-fact_amp,1+fact_amp,batch_size)

        for i in range(batch_size):
            
            # select a random sentence from the list 
            [signal, fs] = sf.read(self.data_list[i])

            # accesing to a random chunk
            snt_len=signal.shape[0]
            snt_beg=np.random.randint(snt_len-wlen-1) #randint(0, snt_len-2*wlen-1)
            snt_end=snt_beg+wlen

            channels = len(signal.shape)
            if channels == 2:
                print('WARNING: stereo to mono: '+self.data_list[i])
                signal = signal[:,0]
            
            sig_batch[i,:]=signal[snt_beg:snt_end]*rand_amp_arr[i]
            lab_batch[i] = self.labels[i]
            
        inp=Variable(torch.from_numpy(sig_batch).float().cuda().contiguous())
        lab=Variable(torch.from_numpy(lab_batch).float().cuda().contiguous())
        
        return inp,lab 

    def __len__(self):
        return len(self.labels)

    def load_feature(self, path):
        signal, fs = sf.read(path)

        # Amplitude normalization
        signal = signal/np.max(np.abs(signal))

        signal = torch.from_numpy(signal).float().contiguous()

        return signal

    def __getitem__(self, index):
        filename_ref = os.path.join(self.test_path, self.pairs[index][0])
        audio_ref = self.load_feature(filename_ref)

        filename_com = os.path.join(self.test_path, self.pairs[index][1])
        audio_com = self.load_feature(filename_com)

        return audio_ref, audio_com, int(self.labels[index][0])
