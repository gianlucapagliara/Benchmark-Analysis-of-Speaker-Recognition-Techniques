### ----------------------- SIGPROC

# https://github.com/jameslyons/python_speech_features

# This file includes routines for basic signal processing including framing and computing power spectra.
# Author: James Lyons 2012
import decimal

import numpy
import math
import logging


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def rolling_window(a, window, step=1):
    # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


def framesig(sig, frame_len, frame_step, winfunc=lambda x: numpy.ones((x,)), stride_trick=True):
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

    zeros = numpy.zeros((padlen - slen,))
    padsignal = numpy.concatenate((sig, zeros))
    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
            numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = numpy.array(indices, dtype=numpy.int32)
        frames = padsignal[indices]
        win = numpy.tile(winfunc(frame_len), (numframes, 1))

    return frames * win


def deframesig(frames, siglen, frame_len, frame_step, winfunc=lambda x: numpy.ones((x,))):
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
    numframes = numpy.shape(frames)[0]
    assert numpy.shape(frames)[
        1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'

    indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
        numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = numpy.array(indices, dtype=numpy.int32)
    padlen = (numframes - 1) * frame_step + frame_len

    if siglen <= 0:
        siglen = padlen

    rec_signal = numpy.zeros((padlen,))
    window_correction = numpy.zeros((padlen,))
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
    if numpy.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            numpy.shape(frames)[1], NFFT)
    complex_spec = numpy.fft.rfft(frames, NFFT)
    return numpy.absolute(complex_spec)


def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0 / NFFT * numpy.square(magspec(frames, NFFT))


def logpowspec(frames, NFFT, norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 0.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the log power spectrum of the corresponding frame.
    """
    ps = powspec(frames, NFFT)
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * numpy.log10(ps)
    if norm:
        return lps - numpy.max(lps)
    else:
        return lps


def preemphasis(signal, coeff=0.95):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return numpy.append(signal[0], signal[1:] - coeff * signal[:-1])


### ----------------------- UTILS

import os
import numpy as np
import librosa
from scipy.signal import lfilter#, butter
# import sigproc

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


def load_wav(filename, sample_rate):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.flatten()
    return audio

# https://github.com/christianvazquez7/ivector/blob/master/MSRIT/rm_dc_n_dither.m

def remove_dc_and_dither(sin, sample_rate):
    if sample_rate == 16e3:
        alpha = 0.99
    elif sample_rate == 8e3:
        alpha = 0.999
    else:
        print("Sample rate must be 16kHz or 8kHz only")
        exit(1)
    sin = lfilter([1,-1], [1,-alpha], sin)
    dither = np.random.random_sample(len(sin)) + np.random.random_sample(len(sin)) - 1
    spow = np.std(dither)
    sout = sin + 1e-6 * spow * dither
    return sout

def normalize_frames(m,epsilon=1e-12):
    return np.array([(v - np.mean(v)) / max(np.std(v),epsilon) for v in m])


def get_fft_spectrum(filename, buckets, sample_rate,n_fft,frame_len,frame_step,preemphasis_alpha):
    signal = load_wav(filename,sample_rate)
    signal *= 2**15
    
    min_len=int(frame_step*sample_rate*list(buckets.keys())[0]+frame_len*sample_rate)            
    if signal.size < min_len:
        signal=np.pad(signal, (0,min_len-signal.size), 'constant',constant_values=0) 

    # get FFT spectrum
    signal = remove_dc_and_dither(signal, sample_rate)
    signal = preemphasis(signal, coeff=preemphasis_alpha)
    frames = framesig(signal, frame_len=frame_len*sample_rate, frame_step=frame_step*sample_rate, winfunc=np.hamming)
    fft = abs(np.fft.fft(frames,n=n_fft))
    fft_norm = normalize_frames(fft.T)

    # truncate to max bucket sizes
    rsize = max(k for k in buckets if k <= fft_norm.shape[1])
    rstart = int((fft_norm.shape[1]-rsize)/2)
    out = fft_norm[:,rstart:rstart+rsize]
    
    return out

def get_fft_spectrum_from_array(sig, buckets, sample_rate,n_fft,frame_len,frame_step,preemphasis_alpha):
    #signal = load_wav(filename,sample_rate)
    signal = sig*(2**15)
    
    min_len=int(frame_step*sample_rate*list(buckets.keys())[0]+frame_len*sample_rate)            
    if signal.size < min_len:
        signal=np.pad(signal, (0,min_len-signal.size), 'constant',constant_values=0) 

    # get FFT spectrum
    signal = remove_dc_and_dither(signal, sample_rate)
    signal = preemphasis(signal, coeff=preemphasis_alpha)
    frames = framesig(signal, frame_len=frame_len*sample_rate, frame_step=frame_step*sample_rate, winfunc=np.hamming)
    fft = abs(np.fft.fft(frames,n=n_fft))
    fft_norm = normalize_frames(fft.T)

    # truncate to max bucket sizes
    #print(fft_norm.shape)
    rsize = max(k for k in buckets if k <= fft_norm.shape[1])
    rstart = int((fft_norm.shape[1]-rsize)/2)
    out = fft_norm[:,rstart:rstart+rsize]
    
    return out
