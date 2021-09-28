import torch
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
    
    def load_state_dict(self, state_dict):
        self_state = self.state_dict()

        for name, param in state_dict.items():
            origname = name
            if name not in self_state:
                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue

            if self_state[name].size() != param.size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(
                    origname, self_state[name].size(), param.size()))
                continue

            self_state[name].copy_(param)

        super(BaseModel, self).load_state_dict(self_state)

    def preforward(self, x):
        x = x.to(self.device)
        return x

    def forward(self, x):
        x = self.preforward(x)
        x = super(BaseModel, self).forward(self, x)
        return x

    def forward_loss(self, x, label, loss, **kwargs):
        x = self.forward(x)
        nloss, prec1 = loss.forward(x, label)

        return nloss, prec1

    def get_feat(self, ref, normalize=True):
        ref_feat = self(ref).to(self.device)
        ref_feat = F.normalize(ref_feat, p=2, dim=1) if normalize else ref_feat

        return ref_feat

class VoxModel(BaseModel):
    def preforward(self, x):
        x = x.reshape(-1, x.size()[-1]).to(self.device)
        return x

    def forward_loss(self, x, label, loss, nPerSpeaker, **kwargs): # TODO: improve implementation
        x = self.forward(x)
        x = x.reshape(nPerSpeaker, -1,
                            x.size()[-1]).transpose(1, 0).squeeze(1)

        nloss, prec1 = loss.forward(x, label)

        return nloss, prec1
