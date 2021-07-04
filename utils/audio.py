import librosa
from typing import Optional, Union
from pathlib import Path
import numpy as np
import random
import soundfile

from tqdm import tqdm

from utils.params import *

def loadWAV(filename, max_frames, evalmode=True, num_eval=10):

    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0, audiosize-max_audio, num=num_eval)
    else:
        startframe = np.array(
            [np.int64(random.random()*(audiosize-max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = np.stack(feats, axis=0).astype(np.float)

    return feat

# AutoSpeech audio functions

int16_max = (2 ** 15) - 1

def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None):
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(fpath_or_wav, sr=None)
    else:
        wav = fpath_or_wav

    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, source_sr, sampling_rate)

    # Apply the preprocessing: normalize volume and shorten long silences
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)

    return wav


def wav_to_spectrogram(wav):
    frames = np.abs(librosa.core.stft(
        wav,
        n_fft=n_fft,
        hop_length=int(sampling_rate * window_step / 1000),
        win_length=int(sampling_rate * window_length / 1000),
    ))
    return frames.astype(np.float32).T


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))

# Auto Speech utils functions

class Utterance:
    def __init__(self, frames_fpath):
        self.frames_fpath = frames_fpath

    def get_frames(self):
        return np.load(self.frames_fpath)

    def random_partial(self, n_frames):
        """
        Crops the frames into a partial utterance of n_frames
        
        :param n_frames: The number of frames of the partial utterance
        :return: the partial utterance frames and a tuple indicating the start and end of the 
        partial utterance in the complete utterance.
        """
        frames = self.get_frames()
        if frames.shape[0] == n_frames:
            start = 0
        else:
            start = np.random.randint(0, frames.shape[0] - n_frames)
        end = start + n_frames
        return frames[start:end], (start, end)

# Contains the set of utterances of a single speaker
class Speaker:
    def __init__(self, root: Path, partition=None):
        self.root = root
        self.partition = partition
        self.name = root.name
        self.utterances = None
        self.utterance_cycler = None
        if self.partition is None:
            with self.root.joinpath("_sources.txt").open("r") as sources_file:
                sources = [l.strip().split(",") for l in sources_file]
        else:
            with self.root.joinpath("_sources_{}.txt".format(self.partition)).open("r") as sources_file:
                sources = [l.strip().split(",") for l in sources_file]
        self.sources = [[self.root, frames_fname, self.name, wav_path]
                        for frames_fname, wav_path in sources]

    def _load_utterances(self):
        self.utterances = [Utterance(source[0].joinpath(source[1]))
                           for source in self.sources]

    def random_partial(self, count, n_frames):
        """
        Samples a batch of <count> unique partial utterances from the disk in a way that all
        utterances come up at least once every two cycles and in a random order every time.

        :param count: The number of partial utterances to sample from the set of utterances from
        that speaker. Utterances are guaranteed not to be repeated if <count> is not larger than
        the number of utterances available.
        :param n_frames: The number of frames in the partial utterance.
        :return: A list of tuples (utterance, frames, range) where utterance is an Utterance,
        frames are the frames of the partial utterances and range is the range of the partial
        utterance with regard to the complete utterance.
        """
        if self.utterances is None:
            self._load_utterances()

        utterances = self.utterance_cycler.sample(count)

        a = [(u,) + u.random_partial(n_frames) for u in utterances]

        return a


def compute_mean_std(dataset_dirs: list, output_path_mean: Path, output_path_std: Path):
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
