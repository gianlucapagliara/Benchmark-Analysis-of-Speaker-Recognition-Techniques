from multiprocess.pool import ThreadPool
import subprocess
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np

import utils.audio as audio


anglophone_nationalites = ["australia", "canada", "ireland", "uk", "usa"]


class DatasetLog:
    """
    Registers metadata about the dataset in a text file.
    """

    def __init__(self, root, name):
        self.text_file = open(Path(root, "Log_%s.txt" %
                              name.replace("/", "_")), "w")
        self.sample_data = dict()

        start_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("Creating dataset %s on %s" % (name, start_time))
        self.write_line("-----")
        self._log_params()

    def _log_params(self):
        self.write_line("Parameter values:")
        for param_name in (p for p in dir(params) if not p.startswith("__")):
            value = getattr(params, param_name)
            self.write_line("\t%s: %s" % (param_name, value))
        self.write_line("-----")

    def write_line(self, line):
        self.text_file.write("%s\n" % line)

    def add_sample(self, **kwargs):
        for param_name, value in kwargs.items():
            if not param_name in self.sample_data:
                self.sample_data[param_name] = []
            self.sample_data[param_name].append(value)

    def finalize(self):
        self.write_line("Statistics:")
        for param_name, values in self.sample_data.items():
            self.write_line("\t%s:" % param_name)
            self.write_line("\t\tmin %.3f, max %.3f" %
                            (np.min(values), np.max(values)))
            self.write_line("\t\tmean %.3f, median %.3f" %
                            (np.mean(values), np.median(values)))
        self.write_line("-----")
        end_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("Finished on %s" % end_time)
        self.text_file.close()


def _init_preprocess_dataset(dataset_name, dataset_root, out_dir):
    if not dataset_root.exists():
        print("Couldn\'t find %s, skipping this dataset." % dataset_root)
        return None, None
    return dataset_root, DatasetLog(out_dir, dataset_name)


def _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, extension,
                             skip_existing, logger):
    print("%s: Preprocessing data for %d speakers." %
          (dataset_name, len(speaker_dirs)))

    # Function to preprocess utterances for one speaker
    def preprocess_speaker(speaker_dir: Path):
        # Give a name to the speaker that includes its dataset
        speaker_name = speaker_dir.parts[-1]

        # Create an output directory with that name, as well as a txt file containing a
        # reference to each source file.
        speaker_out_dir = out_dir.joinpath(speaker_name)
        speaker_out_dir.mkdir(exist_ok=True)
        sources_fpath = speaker_out_dir.joinpath("_sources.txt")

        # There's a possibility that the preprocessing was interrupted earlier, check if
        # there already is a sources file.
        if sources_fpath.exists():
            try:
                with sources_fpath.open("r") as sources_file:
                    existing_fnames = {line.split(
                        ",")[0] for line in sources_file}
            except:
                existing_fnames = {}
        else:
            existing_fnames = {}

        # Gather all audio files for that speaker recursively
        sources_file = sources_fpath.open("a" if skip_existing else "w")
        for in_fpath in speaker_dir.glob("**/*.%s" % extension):
            # Check if the target output file already exists
            out_fname = "_".join(in_fpath.relative_to(speaker_dir).parts)
            out_fname = out_fname.replace(".%s" % extension, ".npy")
            if skip_existing and out_fname in existing_fnames:
                continue

            # Load and preprocess the waveform
            wav = audio.preprocess_wav(in_fpath)
            if len(wav) == 0:
                print(in_fpath)
                continue

            # Create the mel spectrogram, discard those that are too short
            # frames = audio.wav_to_mel_spectrogram(wav)
            frames = audio.wav_to_spectrogram(wav)
            if len(frames) < partials_n_frames:
                continue

            out_fpath = speaker_out_dir.joinpath(out_fname)
            np.save(out_fpath, frames)
            logger.add_sample(duration=len(wav) / sampling_rate)
            sources_file.write("%s,%s\n" % (out_fname, in_fpath))

        sources_file.close()

    # Process the utterances for each speaker
    preprocess_speaker(speaker_dirs[0])
    with ThreadPool(1) as pool:
        list(tqdm(pool.imap(preprocess_speaker, speaker_dirs), dataset_name, len(speaker_dirs),
                  unit="speakers"))
    logger.finalize()
    print("Done preprocessing %s.\n" % dataset_name)


def _preprocess_voxceleb1(dataset_root: Path, partition: str, out_dir: Path, skip_existing=False, filter_anglophones=False):
    # Initialize the preprocessing
    dataset_name = "VoxCeleb1"
    dataset_root, logger = _init_preprocess_dataset(
        dataset_name, dataset_root, out_dir)
    if not dataset_root:
        return

    # Get the contents of the meta file
    with dataset_root.joinpath("vox1_meta.csv").open("r") as metafile:
        metadata = [line.split("\t") for line in metafile][1:]

    # Select the ID and the nationality, filter out non-anglophone speakers
    nationalities = {line[0]: line[3] for line in metadata}
    keep_speaker_ids = [speaker_id for speaker_id, nationality in nationalities.items() if
                        nationality.lower() in anglophone_nationalites]
    print("VoxCeleb1: using samples from %d (presumed anglophone) speakers out of %d." %
          (len(keep_speaker_ids), len(nationalities)))

    # Get the speaker directories for anglophone speakers only
    speaker_dirs = dataset_root.joinpath(partition, 'wav').glob("*")
    speaker_dirs = [speaker_dir for speaker_dir in speaker_dirs]

    print("VoxCeleb1: found %d anglophone speakers on the disk." %
          (len(speaker_dirs)))
    # Preprocess all speakers
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, dataset_root, out_dir, "wav",
                             skip_existing, logger)


def preprocess(dataset_root: Path, skip_existing=True, dev=True, test=True):
    # Process the arguments
    assert dataset_root.exists()

    dev_out_dir = dataset_root.joinpath("feature", "dev")
    dev_out_dir.mkdir(exist_ok=True, parents=True)
    test_out_dir = dataset_root.joinpath("feature", "test")
    test_out_dir.mkdir(exist_ok=True, parents=True)

    merged_out_dirs = [dev_out_dir, test_out_dir]

    # Preprocess the datasets
    if dev:
        _preprocess_voxceleb1(dataset_root, 'dev',
                            dev_out_dir, skip_existing)
    
    if test:
        _preprocess_voxceleb1(dataset_root, 'test',
                            test_out_dir, skip_existing)

    audio.compute_mean_std(merged_out_dirs, dataset_root.joinpath('mean.npy'),
                     dataset_root.joinpath('std.npy'))
