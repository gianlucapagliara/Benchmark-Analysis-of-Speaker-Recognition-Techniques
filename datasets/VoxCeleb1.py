import os
import numpy as np

import torch
from torch.utils.data import Dataset

from utils.audio import loadWAV


class VoxCeleb1(Dataset):
    def __init__(self, test_list, test_path, eval_frames, num_eval, **kwargs):
        self.max_frames = eval_frames
        self.num_eval = num_eval
        self.test_path = test_path
        self.test_list = test_list

        # Read files
        with open(test_list) as dataset_file:
            lines = dataset_file.readlines()

        ## Get a list of unique file names
        files = sum([x.strip().split()[-2:] for x in lines], [])
        self.data_list = list(set(files))
        self.data_list.sort()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        filename = os.path.join(self.test_path, self.data_list[index])
        audio = loadWAV(filename, self.max_frames,
                        evalmode=True, num_eval=self.num_eval)
        return torch.FloatTensor(audio), self.data_list[index]
