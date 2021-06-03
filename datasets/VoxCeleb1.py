import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset

from utils.audio import loadWAV


class VoxCeleb1(Dataset):
    def __init__(self, test_list, test_path, eval_frames, num_eval, max_files=0, **kwargs):
        self.max_frames = eval_frames
        self.num_eval = num_eval
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

    def __getitem__(self, index):
        filename_ref = os.path.join(self.test_path, self.pairs[index][0])
        audio_ref = loadWAV(filename_ref, self.max_frames,
                        evalmode=True, num_eval=self.num_eval)
        
        filename_com = os.path.join(self.test_path, self.pairs[index][1])
        audio_com = loadWAV(filename_com, self.max_frames,
                            evalmode=True, num_eval=self.num_eval)
        
        return torch.FloatTensor(audio_ref), torch.FloatTensor(audio_com), self.labels[index]
