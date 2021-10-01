import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from graphs.models.base import BaseModel, VoxModel


class ConvBnPool(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size, conv_strides, conv_pad,
                 pool_type='', pool_size=(2, 2), pool_strides=None, layer_idx='', conv_layer_prefix='conv'):
        super(ConvBnPool, self).__init__()

        self.layer_idx = layer_idx
        self.conv_layer_prefix = conv_layer_prefix

        self.add_module(f'pad{layer_idx}', nn.ZeroPad2d(padding=conv_pad))
        self.add_module(f'{conv_layer_prefix}{layer_idx}', nn.Conv2d(
            in_channels, out_channels, kernel_size=conv_kernel_size, stride=conv_strides, padding='valid'))
        self.add_module(f'norm{layer_idx}', nn.BatchNorm2d(
            num_features=out_channels, eps=1e-5, momentum=1))
        self.add_module(f'relu{layer_idx}', nn.ReLU())
        if pool_type == 'max':
            self.add_module(f'pool{layer_idx}', nn.MaxPool2d(
                kernel_size=pool_size, stride=pool_strides))
        elif pool_type == 'avg':
            self.add_module(f'pool{layer_idx}', nn.AvgPool2d(
                kernel_size=pool_size, stride=pool_strides))

    def forward(self, x):
        x = self.get_submodule(f'pad{self.layer_idx}')(x)
        x = self.get_submodule(f'{self.conv_layer_prefix}{self.layer_idx}')(x)
        x = self.get_submodule(f'norm{self.layer_idx}')(x)
        x = self.get_submodule(f'relu{self.layer_idx}')(x)
        try:
            x = self.get_submodule(f'pool{self.layer_idx}')(x)
        except:
            pass

        return x


class ConvBnDynamicAPool(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size, conv_strides, conv_pad, layer_idx='', conv_layer_prefix='conv'):
        super(ConvBnDynamicAPool, self).__init__()
        self.layer_idx = layer_idx
        self.conv_layer_prefix = conv_layer_prefix

        self.add_module(f'pad{self.layer_idx}', nn.ZeroPad2d(padding=conv_pad))
        self.add_module(f'{self.conv_layer_prefix}{self.layer_idx}', nn.Conv2d(
            in_channels, out_channels, kernel_size=conv_kernel_size, stride=conv_strides, padding='valid'))
        self.add_module(f'norm{self.layer_idx}', nn.BatchNorm2d(
            num_features=out_channels, eps=1e-5, momentum=1))
        self.add_module(f'relu{self.layer_idx}', nn.ReLU())

    def forward(self, x):
        x = self.get_submodule(f'pad{self.layer_idx}')(x)
        x = self.get_submodule(f'{self.conv_layer_prefix}{self.layer_idx}')(x)
        x = self.get_submodule(f'norm{self.layer_idx}')(x)
        x = self.get_submodule(f'relu{self.layer_idx}')(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))

        return x


class VGGVox(BaseModel):
    def __init__(self, device, nOut=1024, **kwargs):
        super(VGGVox, self).__init__(device)

        # Input: (batch_size, channel, FFT, width) -> (batch_size, 1, 512, width)

        self.conv1 = ConvBnPool(in_channels=1, out_channels=96, conv_kernel_size=(7, 7), conv_strides=(2, 2), conv_pad=(1, 1, 1, 1),
                                pool_type='max', pool_size=(3, 3), pool_strides=(2, 2))
        self.conv2 = ConvBnPool(in_channels=96, out_channels=256, conv_kernel_size=(5, 5), conv_strides=(2, 2), conv_pad=(1, 1, 1, 1),
                                pool_type='max', pool_size=(3, 3), pool_strides=(2, 2))
        self.conv3 = ConvBnPool(in_channels=256, out_channels=384, conv_kernel_size=(
            3, 3), conv_strides=(1, 1), conv_pad=(1, 1, 1, 1))
        self.conv4 = ConvBnPool(in_channels=384, out_channels=256, conv_kernel_size=(
            3, 3), conv_strides=(1, 1), conv_pad=(1, 1, 1, 1))
        self.conv5 = ConvBnPool(in_channels=256, out_channels=256, conv_kernel_size=(3, 3), conv_strides=(1, 1), conv_pad=(1, 1, 1, 1),
                                pool_type='max', pool_size=(5, 3), pool_strides=(3, 2))
        self.fc6 = ConvBnDynamicAPool(in_channels=256, out_channels=4096, conv_kernel_size=(
            9, 1), conv_strides=(1, 1), conv_pad=(0, 0, 0, 0))
        self.fc7 = ConvBnPool(in_channels=4096, out_channels=1024, conv_kernel_size=(
            1, 1), conv_strides=(1, 1), conv_pad=(0, 0, 0, 0))
        
        self.fc8 = nn.Conv2d(in_channels=1024, out_channels=nOut,
                             kernel_size=(1, 1), stride=(1, 1), padding='valid')

    def preforward(self, x):
        x = super().preforward(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        return x

    def forward(self, x):
        x = self.preforward(x)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        
        x = x.squeeze(-1).squeeze(-1).unsqueeze(0)

        return x


class VGGVoxM(VoxModel):
    def __init__(self, device, nOut=1024, encoder_type='SAP', log_input=True, **kwargs):
        super(VGGVoxM, self).__init__(device)

        self.encoder_type = encoder_type
        self.log_input = log_input

        self.netcnn = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(5, 7),
                      stride=(1, 2), padding=(2, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

            nn.Conv2d(96, 256, kernel_size=(5, 5),
                      stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(256, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(256, 512, kernel_size=(4, 1), padding=(0, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

        )

        if self.encoder_type == "MAX":
            self.encoder = nn.AdaptiveMaxPool2d((1, 1))
            out_dim = 512
        elif self.encoder_type == "TAP":
            self.encoder = nn.AdaptiveAvgPool2d((1, 1))
            out_dim = 512
        elif self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(512, 512)
            self.attention = self.new_parameter(512, 1)
            out_dim = 512
        else:
            raise ValueError('Undefined encoder')

        self.fc = nn.Linear(out_dim, nOut)

        self.instancenorm = nn.InstanceNorm1d(40)
        self.torchfb = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=40).to(self.device)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x):

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = x.to(self.device)
                x = self.torchfb(x)+1e-6
                if self.log_input:
                    x = x.log()
                x = self.instancenorm(x).unsqueeze(1)

        x = self.netcnn(x)

        if self.encoder_type == "MAX" or self.encoder_type == "TAP":
            x = self.encoder(x)
            x = x.view((x.size()[0], -1))

        elif self.encoder_type == "SAP":
            x = x.permute(0, 2, 1, 3)
            x = x.squeeze(dim=1).permute(0, 2, 1)  # batch * L * D
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            x = torch.sum(x * w, dim=1)

        x = self.fc(x)

        return x
