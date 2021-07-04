import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
class SpeakerNet(nn.Module):

    def __init__(self, model, loss, device, nPerSpeaker=1, **kwargs):
        super(SpeakerNet, self).__init__()

        self.__S__ = model
        self.__L__ = loss
        self.nPerSpeaker = nPerSpeaker
        self.device = device

    def forward(self, data, label=None):
        # comment for AutoSpeech
        data = data.reshape(-1, data.size()[-1]).to(self.device)
        outp = self.__S__(data)

        if label == None:
            return outp
        else:
            outp = outp.reshape(self.nPerSpeaker, -1,
                                outp.size()[-1]).transpose(1, 0).squeeze(1)  # comment for AutoSpeech

            nloss, prec1 = self.__L__.forward(outp, label)

            return nloss, prec1
'''

class BaseModel(nn.Module):
    """
    This base class will contain the base functions to be overloaded by any model you will implement.
    """

    def __init__(self, device):
        super(BaseModel, self).__init__()
        self.device = device
        self.to(self.device)

    def forward(self, input):
        return nn.Module.forward(self, input)

    def scoring(self, ref, com):
        raise NotImplementedError


class AutoSpeechModel(BaseModel):
    def __init__(self, device):
        super(AutoSpeechModel, self).__init__(device)

    def forward(self, input):
        return BaseModel.forward(self, input)

    def scoring(self, ref, com):
        # Data shape preparation
        ref = ref.to(self.device).squeeze(0)
        com = com.to(self.device).squeeze(0)

        # Feature extraction
        ref_feat = self(ref).to(self.device)
        com_feat = self(com).to(self.device)

        ref_feat = ref_feat.mean(dim=0).unsqueeze(0)
        com_feat = com_feat.mean(dim=0).unsqueeze(0)

        # Distance
        # score = F.cosine_similarity(ref_feat, com_feat)
        score = F.pairwise_distance(ref_feat, com_feat)
        score = -1 * score.data.cpu().numpy()[0]

        return score


class VoxModel(BaseModel):
    def __init__(self, device):
        super(VoxModel, self).__init__(device)

    def forward(self, input):
        input = input.reshape(-1, input.size()[-1]).to(self.device)
        outp = BaseModel.forward(self, input)

        return outp

    def forward(self, input, label):  # TODO: fix -> derived from SpeakerNet
        outp = outp.reshape(self.nPerSpeaker, -1,
                            outp.size()[-1]).transpose(1, 0).squeeze(1)

        nloss, prec1 = self.__L__.forward(outp, label)

        return nloss, prec1

    def scoring(self, ref, com):
        # Feature extraction
        ref_feat = self(ref).to(self.device)
        com_feat = self(com).to(self.device)

        # Normalization
        if self.__L__.test_normalize:
            ref_feat = F.normalize(ref_feat, p=2, dim=1)
            com_feat = F.normalize(com_feat, p=2, dim=1)

        # is it usefull?
        ref_feat = ref_feat.unsqueeze(-1)
        com_feat = com_feat.unsqueeze(-1).transpose(0, 2)

        # Distance
        score = F.pairwise_distance(ref_feat, com_feat)
        score = -1 * score.data.cpu().numpy()[0]

        return score
