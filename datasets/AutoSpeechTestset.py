import os
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms as T
from pathlib import Path

from utils.transforms import Normalize, generate_test_sequence
from utils.params import *


def get_test_paths(pairs_path, db_dir):
    def convert_folder_name(path):
        basename = os.path.splitext(path)[0]
        items = basename.split('/')
        speaker_dir = items[0]
        fname = '{}_{}.npy'.format(items[1], items[2])
        p = os.path.join(speaker_dir, fname)
        return p

    pairs = [line.strip().split()
             for line in open(pairs_path, 'r').readlines()]
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []

    for pair in pairs:
        if pair[0] == '1':
            issame = True
        else:
            issame = False

        path0 = db_dir.joinpath(convert_folder_name(pair[1]))
        path1 = db_dir.joinpath(convert_folder_name(pair[2]))

        # Only add the pair if both paths exist
        if os.path.exists(path0) and os.path.exists(path1):
            path_list.append((path0, path1, issame))
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list


class AutoSpeechTestset(Dataset):
    def __init__(self, test_list, test_path, partial_n_frames, **kwargs):
        super(AutoSpeechTestset, self).__init__()
        self.test_path = Path(test_path)
        self.root = self.test_path.joinpath('feature', 'test')
        self.test_pair_txt_fpath = Path(test_list)
        self.test_pairs = get_test_paths(self.test_pair_txt_fpath, self.root)
        self.partial_n_frames = partial_n_frames
        mean = np.load(self.test_path.joinpath('mean.npy'))
        std = np.load(self.test_path.joinpath('std.npy'))
        self.transform = T.Compose([
            Normalize(mean, std)
        ])

    def load_feature(self, feature_path):
        feature = np.load(feature_path)
        test_sequence = generate_test_sequence(feature, self.partial_n_frames)

        test_sequence = test_sequence[0, :, :]

        if self.transform is not None:
            test_sequence = self.transform(test_sequence)

        return test_sequence

    def __getitem__(self, index):
        (path_1, path_2, issame) = self.test_pairs[index]

        feature1 = self.load_feature(path_1)
        feature2 = self.load_feature(path_2)

        return feature1, feature2, issame

    def __len__(self):
        return len(self.test_pairs)
