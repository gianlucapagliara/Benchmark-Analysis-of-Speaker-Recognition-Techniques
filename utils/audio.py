import librosa
from typing import Optional, Union
from pathlib import Path
import numpy as np
import speechpy
import math
import logging
import decimal
from scipy.signal import lfilter

from tqdm import tqdm

from utils.Speaker import Speaker

INT16_MAX_VALUE = (2 ** 15) - 1

def load_wav(fpath, source_sr = None):
    wav, source_sr = librosa.load(fpath, sr=source_sr, mono=True)
    wav = wav.flatten()

    return (wav, source_sr)


def wav_to_spectrogram(wav, n_fft, sampling_rate, window_step, window_length):
    frames = np.abs(librosa.core.stft(
        wav,
        n_fft=n_fft,
        hop_length=int(sampling_rate * window_step / 1000),
        win_length=int(sampling_rate * window_length / 1000),
    ))
    return frames.astype(np.float32).T

def normalize_volume(wav, target_dBFS = -30, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * INT16_MAX_VALUE) ** 2))
    wave_dBFS = 20 * np.log10(rms / INT16_MAX_VALUE)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))


def generate_test_sequence(feature, partial_n_frames, shift=None):
    while feature.shape[0] <= partial_n_frames:
        feature = np.repeat(feature, 2, axis=0)
    if shift is None:
        shift = partial_n_frames // 2
    test_sequence = []
    start = 0
    while start + partial_n_frames <= feature.shape[0]:
        test_sequence.append(feature[start: start + partial_n_frames])
        start += shift
    test_sequence = np.stack(test_sequence, axis=0)
    return test_sequence

def compute_mean_std(dataset_dirs, output_path_mean, output_path_std):
    print("Computing mean std...")

    speaker_dirs = []
    for dd in dataset_dirs:
        speaker_dirs.extend([f for f in dd.glob("*") if f.is_dir()])
    
    if len(speaker_dirs) == 0:
        raise Exception("No speakers found. Make sure you are pointing to the directory "
                        "containing all preprocessed speaker directories.")
    speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs]

    sources = []
    for speaker in speakers:
        sources.extend(speaker.sources)

    sumx = np.zeros(257, dtype=np.float32)
    sumx2 = np.zeros(257, dtype=np.float32)
    count = 0
    n = len(sources)
    for i, source in tqdm(enumerate(sources), total=n):
        feature = np.load(source[0].joinpath(source[1]))
        sumx += feature.sum(axis=0)
        sumx2 += (feature * feature).sum(axis=0)
        count += feature.shape[0]

    mean = sumx / count
    std = np.sqrt(sumx2 / count - mean * mean)

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    np.save(output_path_mean, mean)
    np.save(output_path_std, std)

def get_logenergy(signal, fs, num_coefficient=40, fft_length=1024, frame_length=0.025, frame_stride=0.01):
    # Staching frames
    frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=frame_length,
                                              frame_stride=frame_stride,
                                              zero_padding=True)

    # # Extracting power spectrum (choosing 3 seconds and elimination of DC)
    power_spectrum = speechpy.processing.power_spectrum(
        frames, fft_points=2 * num_coefficient)[:, 1:]

    logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=frame_length, frame_stride=frame_stride,
                                      num_filters=num_coefficient, fft_length=fft_length, low_frequency=0,
                                      high_frequency=None)

    return logenergy


# https://github.com/jameslyons/python_speech_features
def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def rolling_window(a, window, step=1):
    # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


def framesig(sig, frame_len, frame_step, winfunc=lambda x: np.ones((x,)), stride_trick=True):
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + \
            int(math.ceil((1.0 * slen - frame_len) / frame_step))  # LV

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))
    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
            np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = np.array(indices, dtype=np.int32)
        frames = padsignal[indices]
        win = np.tile(winfunc(frame_len), (numframes, 1))

    return frames * win


def deframesig(frames, siglen, frame_len, frame_step, winfunc=lambda x: np.ones((x,))):
    """Does overlap-add procedure to undo the action of framesig.

    :param frames: the array of frames.
    :param siglen: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: a 1-D signal.
    """
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = np.shape(frames)[0]
    assert np.shape(frames)[
        1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'

    indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
        np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = np.array(indices, dtype=np.int32)
    padlen = (numframes - 1) * frame_step + frame_len

    if siglen <= 0:
        siglen = padlen

    rec_signal = np.zeros((padlen,))
    window_correction = np.zeros((padlen,))
    win = winfunc(frame_len)

    for i in range(0, numframes):
        window_correction[indices[i, :]] = window_correction[
            indices[i, :]] + win + 1e-15  # add a little bit so it is never zero
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

    rec_signal = rec_signal / window_correction
    return rec_signal[0:siglen]


def magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    if np.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            np.shape(frames)[1], NFFT)
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)


def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0 / NFFT * np.square(magspec(frames, NFFT))


def logpowspec(frames, NFFT, norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 0.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the log power spectrum of the corresponding frame.
    """
    ps = powspec(frames, NFFT)
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * np.log10(ps)
    if norm:
        return lps - np.max(lps)
    else:
        return lps


def preemphasis(signal, coeff=0.95):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

# https://github.com/christianvazquez7/ivector/blob/master/MSRIT/rm_dc_n_dither.m
def remove_dc_and_dither(sin, sample_rate):
    if sample_rate == 16e3:
        alpha = 0.99
    elif sample_rate == 8e3:
        alpha = 0.999
    else:
        print("Sample rate must be 16kHz or 8kHz only")
        exit(1)
    sin = lfilter([1, -1], [1, -alpha], sin)
    dither = np.random.random_sample(
        len(sin)) + np.random.random_sample(len(sin)) - 1
    spow = np.std(dither)
    sout = sin + 1e-6 * spow * dither
    return sout


def normalize_frames(m, epsilon=1e-12):
    return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in m])


def get_fft_spectrum(signal, buckets, sample_rate, n_fft, frame_len, frame_step, preemphasis_alpha):
    signal *= 2**15

    min_len = int(frame_step*sample_rate*list(buckets.keys())
                  [0]+frame_len*sample_rate)
    if signal.size < min_len:
        signal = np.pad(signal, (0, min_len-signal.size),
                        'constant', constant_values=0)

    # get FFT spectrum
    signal = remove_dc_and_dither(signal, sample_rate)
    signal = preemphasis(signal, coeff=preemphasis_alpha)
    frames = framesig(signal, frame_len=frame_len*sample_rate,
                      frame_step=frame_step*sample_rate, winfunc=np.hamming)
    fft = abs(np.fft.fft(frames, n=n_fft))
    fft_norm = normalize_frames(fft.T)

    # truncate to max bucket sizes
    rsize = max(k for k in buckets if k <= fft_norm.shape[1])
    rstart = int((fft_norm.shape[1]-rsize)/2)
    out = fft_norm[:, rstart:rstart+rsize]

    return out

def build_buckets(max_sec, step_sec, frame_step):
    buckets = {}
    frames_per_sec = int(1/frame_step)
    end_frame = int(max_sec*frames_per_sec)
    step_frame = int(step_sec*frames_per_sec)
    for i in range(0, end_frame+1, step_frame):
        s = i
        s = np.floor((s-7+2)/2) + 1  # conv1
        s = np.floor((s-3)/2) + 1  # mpool1
        s = np.floor((s-5+2)/2) + 1  # conv2
        s = np.floor((s-3)/2) + 1  # mpool2
        s = np.floor((s-3+2)/1) + 1  # conv3
        s = np.floor((s-3+2)/1) + 1  # conv4
        s = np.floor((s-3+2)/1) + 1  # conv5
        s = np.floor((s-3)/2) + 1  # mpool5
        s = np.floor((s-1)/1) + 1  # fc6
        if s > 0:
            buckets[i] = int(s)
    return buckets


def get_cube(feature, cube_shape):
    num_frames = cube_shape[0]
    num_coefficient = cube_shape[1]
    num_utterances = cube_shape[2]

    # Feature cube.
    feature_cube = np.zeros(
        (num_utterances, num_frames, num_coefficient), dtype=np.float32)

    # Get some random starting point for creation of the future cube of size (num_frames x num_coefficient x num_utterances)
    # Since we are doing random indexing, the data augmentation is done as well because in each iteration it returns another indexing!
    idx = np.random.randint(
        feature.shape[0] - num_frames, size=num_utterances)
    for num, index in enumerate(idx):
        feature_cube[num, :, :] = feature[index:index + num_frames, :]

    return feature_cube[None, :, :, :]
