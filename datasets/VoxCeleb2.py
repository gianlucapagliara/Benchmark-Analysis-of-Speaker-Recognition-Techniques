import os
import numpy as np

import torch
from torch.utils.data import Dataset

from utils.audio import load_wav

class VoxCeleb2(Dataset):
    def __init__(self, train_list, train_path, max_frames, max_files=0, **kwargs):
        self.train_list = train_list
        self.max_frames = max_frames

        # Read training files
        with open(train_list) as dataset_file:
            lines = dataset_file.readlines()

        # Make a dictionary of ID names and ID indices (i.e. {'id0001': 0})
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: i for i, key in enumerate(dictkeys)}

        # Parse the training list into filenames and ID indices
        self.data_list = []
        self.data_label = []

        for lidx, line in enumerate(lines):
            data = line.strip().split()

            speaker_label = dictkeys[data[0]]
            filename = os.path.join(train_path, data[1])

            self.data_label.append(speaker_label)
            self.data_list.append(filename)

        if max_files > 0:
            self.data_label = self.data_label[:max_files]
            self.data_list = self.data_list[:max_files]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, indices):
        feat = []

        if hasattr(indices, '__iter__'):
            for index in indices:
                audio, sr = load_wav(self.data_list[index])
                audio = voxceleb_trainer_preprocessing(audio, sr, self.max_frames, evalmode=False)
                feat.append(audio)
        else:
            index = indices
            audio, sr = load_wav(self.data_list[index])
            audio = voxceleb_trainer_preprocessing(
                audio, sr, self.max_frames, evalmode=False)
            feat.append(audio)
        
        feat = np.concatenate(feat, axis=0)

        return torch.FloatTensor(feat), self.data_label[index]
