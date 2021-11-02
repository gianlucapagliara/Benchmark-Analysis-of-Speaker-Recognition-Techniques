import torch
import numpy as np
from pyannote.audio.features import RawAudio
from pyannote.core import SlidingWindowFeature

from graphs.models.base import BaseModel


class SincNetPyAnnote(BaseModel):
    def __init__(self, device, weights='emb_voxceleb', sample_rate=16000, **kwargs):
        super(SincNetPyAnnote, self).__init__(device)

        self.model = torch.hub.load('pyannote/pyannote-audio', weights)
        self.model.device = self.device
        self.sample_rate = sample_rate

    def forward(self, x):
        # embeddings = torch.zeros(
        #     x.shape[0], 512).float().to(self.device)

        for i in range(0, x.shape[0]):
            # x = x[i].cpu().detach().numpy()
            x = x.squeeze(0).cpu().detach().numpy()

            feature_extraction_ = RawAudio(sample_rate=self.sample_rate)

            features = SlidingWindowFeature(
                feature_extraction_.get_features(x, self.sample_rate),
                feature_extraction_.sliding_window,
            )

            x = self.model.model_.slide(
                    features,
                    self.model.chunks_,
                    batch_size=self.model.batch_size,
                    device=self.model.device,
                    return_intermediate=self.model.return_intermediate,
                    progress_hook=self.model.progress_hook,
                ).data
            
            x = np.mean(x, axis=0)

            x = torch.FloatTensor(x).unsqueeze(0)

            # embeddings[i] = x
        
        embeddings = x

        return embeddings
