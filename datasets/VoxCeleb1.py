import os

import torch
from torch.utils.data import Dataset

import utils.audio as audio

class VoxCeleb1(Dataset):
    def __init__(self, file_list, base_path, sampling_rate=16000, max_files=0, preprocessing_function=None, **kwargs):
        self.base_path = base_path
        self.file_list = file_list

        self.data_list = []
        self.labels = []
        self.pairs = []

        # Read files
        with open(file_list) as dataset_file:
            lines = dataset_file.readlines()

        for line in lines:
            data = line.split()

            # Append random label if missing
            if len(data) > 2:
                self.labels.append(data[0])
                self.pairs.append([data[1], data[2]])

        if max_files > 0:
            self.pairs = self.pairs[:max_files]
            self.labels = self.labels[:max_files]

        self.sampling_rate = sampling_rate
        self.preprocessing_function = preprocessing_function
        self.kwargs = kwargs

    def __len__(self):
        return len(self.labels)

    def __load_feature__(self, path):
        signal, sr = audio.load_wav(path, source_sr=self.sampling_rate)

        if self.preprocessing_function is not None:
            signal = self.preprocessing_function(signal, sr, **self.kwargs)

        return (signal, sr)

    def __getitem__(self, index):
        filename_ref = os.path.join(self.base_path, self.pairs[index][0])
        audio_ref, _ = self.__load_feature__(filename_ref)

        filename_com = os.path.join(self.base_path, self.pairs[index][1])
        audio_com, _ = self.__load_feature__(filename_com)

        label = int(self.labels[index][0])

        return audio_ref, audio_com, label
