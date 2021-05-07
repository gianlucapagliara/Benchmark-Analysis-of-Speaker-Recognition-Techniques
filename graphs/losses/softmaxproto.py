import torch
import torch.nn as nn
from graphs.losses.softmax import LossFunction as Softmax
from graphs.losses.angleproto import LossFunction as Angleproto

class LossFunction(nn.Module):

    def __init__(self, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True

        self.softmax = Softmax(**kwargs)
        self.angleproto = Angleproto(**kwargs)

    def forward(self, x, label=None):

        assert x.size()[1] == 2

        nlossS, prec1   = self.softmax(x.reshape(-1,x.size()[-1]), label.repeat_interleave(2))

        nlossP, _       = self.angleproto(x,None)

        return nlossS+nlossP, prec1
