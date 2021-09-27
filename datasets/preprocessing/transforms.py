import numpy as np
import torchvision.transforms as transforms
from torchvision import transforms as T
import utils.transforms as transforms

def get_autospeech_transform(partial_n_frames, mean_file, std_file):
    mean = np.load(mean_file)
    std = np.load(std_file)

    transform = T.Compose([
        transforms.WawToSpectogram(),
        transforms.GenerateSequence(partial_n_frames),
        transforms.Normalize(mean, std)
    ])

    return transform


def get_cnn3d_transform(cube_shape=(80, 40, 20)):
    transform = transforms.Compose(
        [transforms.CMVN(), transforms.FeatureCube(cube_shape)])

    return transform
