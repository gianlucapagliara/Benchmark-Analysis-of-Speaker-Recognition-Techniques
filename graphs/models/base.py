import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseModel(nn.Module):
    """
    This base class will contain the base functions to be overloaded by any model you will implement.
    """

    def __init__(self, device):
        super(BaseModel, self).__init__()
        self.device = device
        self = self.to(self.device)

    def preforward(self, x):
        return x

    def forward(self, x):
        x = self.preforward(x)
        x = super(BaseModel, self).forward(self, x)
        return x

    def forward_loss(self, x, label, loss, **kwargs):
        x = self.forward(x)
        nloss, prec1 = loss.forward(x, label)

        return nloss, prec1


    def scoring(self, ref, com, normalize=False):
        # Feature extraction
        ref_feat = self(ref).to(self.device)
        com_feat = self(com).to(self.device)

        # Distance
        score = F.pairwise_distance(ref_feat, com_feat)
        score = score.detach().cpu().numpy()
        score = -1 * np.mean(score)

        return score


class AutoSpeechModel(BaseModel):
    def __init__(self, device):
        super(AutoSpeechModel, self).__init__(device)

    def forward(self, x):
        return super(AutoSpeechModel, self).forward(self, x)

    def scoring(self, ref, com, normalize=False):
        # Data shape preparation
        ref = ref.squeeze(0)
        com = com.squeeze(0)

        # Feature extraction
        ref_feat = self(ref).to(self.device)
        com_feat = self(com).to(self.device)

        ref_feat = ref_feat.mean(dim=0).unsqueeze(0)
        com_feat = com_feat.mean(dim=0).unsqueeze(0)

        # Distance
        score = F.cosine_similarity(ref_feat, com_feat)
        score = score.data.cpu().numpy()[0]
        # score = F.pairwise_distance(ref_feat, com_feat)
        # score = -1 * score.data.cpu().numpy()[0]

        return score


class VoxModel(BaseModel):
    def __init__(self, device):
        super(VoxModel, self).__init__(device)

    def preforward(self, x):
        x = x.reshape(-1, x.size()[-1]).to(self.device)
        return x

    def forward_loss(self, x, label, loss, nPerSpeaker, **kwargs): # TODO: improve implementation
        x = self.forward(x)
        x = x.reshape(nPerSpeaker, -1,
                            x.size()[-1]).transpose(1, 0).squeeze(1)

        nloss, prec1 = loss.forward(x, label)

        return nloss, prec1

    def scoring(self, ref, com, normalize=False):
        # Feature extraction
        ref_feat = self(ref).to(self.device)
        com_feat = self(com).to(self.device)

        # Normalization
        if normalize:
            ref_feat = F.normalize(ref_feat, p=2, dim=1)
            com_feat = F.normalize(com_feat, p=2, dim=1)

        # is it usefull?
        ref_feat = ref_feat.unsqueeze(-1)
        com_feat = com_feat.unsqueeze(-1).transpose(0, 2)

        # Distance
        score = F.pairwise_distance(ref_feat, com_feat)
        score = score.detach().cpu().numpy()
        score = -1 * np.mean(score)

        return score
