import numpy as np
import torch

from utils.misc import round_down


class Sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size, distributed, seed, **kwargs):
        
        self.data_source = data_source
        self.data_label = data_source.data_label
        self.nPerSpeaker = nPerSpeaker
        self.max_seg_per_spk = max_seg_per_spk
        self.batch_size = batch_size
        self.epoch = 0
        self.seed = seed
        self.distributed = distributed

    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_label), generator=g).tolist()

        data_dict = {}

        # Sort into dictionary of file indices for each ID
        for index in indices:
            speaker_label = self.data_label[index]
            if not (speaker_label in data_dict):
                data_dict[speaker_label] = []
            data_dict[speaker_label].append(index)

        ## Group file indices for each class
        dictkeys = list(data_dict.keys())
        dictkeys.sort()

        def lol(lst, sz): return [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        for findex, key in enumerate(dictkeys):
            data = data_dict[key]
            numSeg = round_down(
                min(len(data), self.max_seg_per_spk), self.nPerSpeaker)

            rp = lol(np.arange(numSeg), self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Mix data in random order
        mixid = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel = []
        mixmap = []

        ## Prevent two pairs of the same speaker in the same batch
        for i in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            if flattened_label[i] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[i])
                mixmap.append(i)

        mixed_list = [flattened_list[i] for i in mixmap]

        ## Divide data to each GPU
        if self.distributed:
            total_size = len(
                mixed_list) - len(mixed_list) % (self.batch_size * dist.get_world_size())
            start_index = int((dist.get_rank()) /
                              dist.get_world_size() * total_size)
            end_index = int((dist.get_rank() + 1) /
                            dist.get_world_size() * total_size)
            return iter(mixed_list[start_index:end_index])
        else:
            total_size = len(mixed_list) - len(mixed_list) % self.batch_size
            return iter(mixed_list[:total_size])

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
