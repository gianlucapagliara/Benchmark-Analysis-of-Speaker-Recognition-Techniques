import numpy as np
import random
import torch


import utils.audio as audio
from speechpy.processing import cmvn

from torchvision import transforms as T
from utils.transforms import Normalize


def voxceleb_trainer_preprocessing(signal, sampling_rate, max_frames, evalmode=True, num_eval=10, **kwargs):
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    audiosize = signal.shape[0]

    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        signal = np.pad(signal, (0, shortage), 'wrap')
        audiosize = signal.shape[0]

    if evalmode:
        startframe = np.linspace(0, audiosize-max_audio, num=num_eval)
    else:
        startframe = np.array(
            [np.int64(random.random()*(audiosize-max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(signal)
    else:
        for asf in startframe:
            feats.append(signal[int(asf):int(asf)+max_audio])

    feat = np.stack(feats, axis=0).astype(np.float)

    return feat


def cnn3d_preprocessing(signal, sampling_rate, num_coefficient=40, C=20, **kwargs):
    signal = audio.get_logenergy(
        signal, sampling_rate, num_coefficient=num_coefficient)
    signal = cmvn(signal, variance_normalization=False)
    signal = audio.get_cube(signal, (80, 40, 20))
    signal = torch.FloatTensor(signal)

    return signal


def vggvox_preprocessing(signal, sampling_rate, max_sec=10, buckets=None, bucket_step=1, n_fft=512, frame_len=0.025, frame_step=0.005, preemphasis_alpha=0.97, **kwargs):
    buckets = audio.build_buckets(max_sec, bucket_step,
                                  frame_step) if buckets is None else buckets

    signal = audio.get_fft_spectrum(
        signal, buckets, sampling_rate, n_fft, frame_len, frame_step, preemphasis_alpha)

    return signal


def amplitude_preprocessing(signal, sampling_rate, **kwargs):  # MobileNet
    signal = signal/np.max(np.abs(signal))

    return signal


def sincnet_preprocessing(signal, sampling_rate, wlen, wshift, max_frames=400, **kwargs):
    wlen = int(sampling_rate*wlen/1000.00)
    wshift = int(sampling_rate*wshift/1000.00)

    signal = signal/np.max(np.abs(signal))
    signal = torch.FloatTensor(audio.framesig(signal, wlen, wshift))

    if(signal.shape[0] > max_frames):
        signal = signal[:400]

    return signal


def autospeech_preprocessing(signal, sampling_rate, partial_n_frames, mean_file, std_file, n_fft=512, window_step=10, window_length=25, shift=None, **kwargs):
    signal = audio.normalize_volume(signal, increase_only=True)
    
    signal = audio.wav_to_spectrogram(
        signal, n_fft, sampling_rate, window_step, window_length)
    if len(signal) < partial_n_frames:
        print("Too shoort")

    signal = audio.generate_test_sequence(signal, partial_n_frames, shift=shift)[0, :, :]
    
    mean = np.load(mean_file)
    std = np.load(std_file)
    transform = T.Compose([
        Normalize(mean, std)
    ])
    signal = transform(signal)

    return signal


def pyannote_preprocessing(signal, sampling_rate, **kwargs):
    signal = np.expand_dims(signal, axis=1)
    
    return signal
