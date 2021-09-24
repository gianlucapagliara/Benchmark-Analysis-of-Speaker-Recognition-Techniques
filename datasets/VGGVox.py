import random
import os

import torch
from torch.utils.data import Dataset

from utils.vggvox import *

###PARAMETERS
# Signal processing
SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
#FRAME_LEN = 0.05
FRAME_LEN = 0.025
FRAME_STEP = 0.005
NUM_FFT = 512
BUCKET_STEP = 1
MAX_SEC = 10


class VGGVox(Dataset):
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

        self.buckets = build_buckets(MAX_SEC, BUCKET_STEP, FRAME_STEP)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        filename_ref = os.path.join(self.test_path, self.pairs[index][0])
        audio_ref = get_fft_spectrum(
            filename_ref, self.buckets, SAMPLE_RATE, NUM_FFT, FRAME_LEN, FRAME_STEP, PREEMPHASIS_ALPHA)

        filename_com = os.path.join(self.test_path, self.pairs[index][1])
        audio_com = get_fft_spectrum(
            filename_com, self.buckets, SAMPLE_RATE, NUM_FFT, FRAME_LEN, FRAME_STEP, PREEMPHASIS_ALPHA)

        return torch.FloatTensor(audio_ref), torch.FloatTensor(audio_com), int(self.labels[index][0])
