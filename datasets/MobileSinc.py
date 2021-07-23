import os
import random
import numpy as np
import soundfile as sf

import torch
import torchaudio
from torch.utils.data import Dataset


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
