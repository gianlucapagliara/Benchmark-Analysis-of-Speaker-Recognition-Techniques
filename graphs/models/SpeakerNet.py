import torch.nn as nn


class SpeakerNet(nn.Module):

    def __init__(self, model, loss, device, nPerSpeaker=1, **kwargs):
        super(SpeakerNet, self).__init__()

        self.__model__ = model
        self.__loss__ = loss
        self.nPerSpeaker = nPerSpeaker
        self.device = device

    def forward(self, data, label=None):

        data = data.reshape(-1, data.size()[-1]).to(self.device)
        outp = self.__model__.forward(data)

        if label == None:
            return outp

        else:
            outp = outp.reshape(self.nPerSpeaker, -1,
                                outp.size()[-1]).transpose(1, 0).squeeze(1)

            nloss, prec1 = self.__loss__.forward(outp, label)

            return nloss, prec1
