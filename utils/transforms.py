import numpy as np
import random
import speechpy
import utils.audio as audio

INT16_MAX_VALUE = (2 ** 15) - 1

class Normalize(object):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, input):
        return (input - self.mean) / self.std

class NormalizeVolume(object):
    def __init__(self, target_dBFS=-30, increase_only=False, decrease_only=False):
        super(NormalizeVolume, self).__init__()
        
        if increase_only and decrease_only:
            raise ValueError("Both increase only and decrease only are set")
        
        self.target_dBFS = target_dBFS
        self.increase_only = increase_only
        self.decrease_only = decrease_only

    def __call__(self, wav):
        rms = np.sqrt(np.mean((wav * INT16_MAX_VALUE) ** 2))
        wave_dBFS = 20 * np.log10(rms / INT16_MAX_VALUE)
        dBFS_change = self.target_dBFS - wave_dBFS
        if dBFS_change < 0 and self.increase_only or dBFS_change > 0 and self.decrease_only:
            return wav
        return wav * (10 ** (dBFS_change / 20))


class TimeReverse(object):
    def __init__(self, p=0.5):
        super(TimeReverse, self).__init__()
        self.p = p

    def __call__(self, input):
        if random.random() < self.p:
            return np.flip(input, axis=0).copy()
        return input


class CMVN(object):
    """
    Cepstral mean variance normalization.
    """

    def __call__(self, feature):
        # Mean variance normalization of the spectrum.
        feature_norm = speechpy.processing.cmvn(
            feature, variance_normalization=False)

        return feature_norm


class FeatureCube(object):
    """
    Return a feature cube of desired size.

    Args:
        cube_shape (tuple): The shape of the feature cube.
    """

    def __init__(self, cube_shape):
        assert isinstance(cube_shape, (tuple))
        self.cube_shape = cube_shape
        self.num_frames = cube_shape[0]
        self.num_coefficient = cube_shape[1]
        self.num_utterances = cube_shape[2]

    def __call__(self, feature):
        return audio.get_cube(feature, self.cube_shape)

class LogEnergy(object):
    def __init__(self, sample_rate, num_coefficient=40):
        super(LogEnergy).__init__()
        self.sample_rate = sample_rate
        self.num_coefficient = num_coefficient

    def __call__(self, input):
        return audio.get_logenergy(input, self.sample_rate, num_coefficient=self.num_coefficient)

class GenerateSequence(object):
    def __init__(self, partial_n_frames, shift=None):
        super(GenerateSequence, self).__init__()

        self.partial_n_frames = partial_n_frames
        self.shift = self.partial_n_frames // 2 if shift is None else shift
    
    def __call__(self, input):
        return audio.generate_test_sequence(input, self.partial_n_frames, self.shift)

class WawToSpectogram(object):
    def __init__(self, n_fft, sampling_rate, window_step):
        super(WawToSpectogram).__init__()
        self.n_fft = n_fft
        self.sampling_rate = sampling_rate
        self.window_step = window_step

    def __call__(self, input):
        return audio.wav_to_spectrogram(input, self.n_fft, self.sampling_rate, self.window_step)
