import torch
import torchvision
import torchvision.transforms as transforms
import os
import soundfile as sf
from scipy.io.wavfile import read
import scipy.io.wavfile as wav
import subprocess as sp
import numpy as np
import argparse
import random
import os
import sys
from random import shuffle
import datetime

from utils.audio import get_logenergy, CMVN, Feature_Cube, ToOutput

default_transform = transforms.Compose(
    [CMVN(), Feature_Cube((80, 40, 20)), ToOutput()])


class Dataset3D():
    def __init__(self, test_list, test_path, max_files=0, transform=default_transform, num_coefficient=40, **kwargs):
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

        self.num_coefficient = num_coefficient

        self.transform = transform

    def load_feature(self, path):
        signal, fs = sf.read(path)

        feature = get_logenergy(
            signal, fs, num_coefficient=self.num_coefficient)

        if self.transform:
            feature = self.transform(feature)

        return feature

    def __getitem__(self, index):
        filename_ref = os.path.join(self.test_path, self.pairs[index][0])
        audio_ref = self.load_feature(filename_ref)

        filename_com = os.path.join(self.test_path, self.pairs[index][1])
        audio_com = self.load_feature(filename_com)

        return audio_ref, audio_com, int(self.labels[index][0])

    def __len__(self):
        return len(self.labels)
