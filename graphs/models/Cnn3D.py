import torch.nn as nn
import torch.nn.functional as F

from graphs.models.base import BaseModel

class Cnn3D(BaseModel):
    def __init__(self, device, **kwargs):
        super(Cnn3D, self).__init__(device)
        
        # self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv11 = nn.Conv3d(1, 16, (4, 9, 9), stride=(1, 2, 1))
        self.conv11_bn = nn.BatchNorm3d(16)
        self.conv12 = nn.Conv3d(16, 16, (4, 9, 9), stride=(1, 1, 1))
        self.conv12_bn = nn.BatchNorm3d(16)
        self.conv21 = nn.Conv3d(16, 32, (3, 7, 7), stride=(1, 1, 1))
        self.conv21_bn = nn.BatchNorm3d(32)
        self.conv22 = nn.Conv3d(32, 32, (3, 7, 7), stride=(1, 1, 1))
        self.conv22_bn = nn.BatchNorm3d(32)
        self.conv31 = nn.Conv3d(32, 64, (3, 5, 5), stride=(1, 1, 1))
        self.conv31_bn = nn.BatchNorm3d(64)
        self.conv32 = nn.Conv3d(64, 64, (3, 5, 5), stride=(1, 1, 1))
        self.conv32_bn = nn.BatchNorm3d(64)
        self.conv41 = nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 1, 1))
        self.conv41_bn = nn.BatchNorm3d(128)
        # self.conv42 = nn.Conv3d(128, 512, (4, 6, 2), stride=(1, 1, 1))
        # self.conv42_bn = nn.BatchNorm3d(512)

        # Fully-connected
        self.fc1 = nn.Linear(128 * 4 * 6 * 2, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1211)

    def forward(self, x):
        x = F.relu(self.conv11_bn(self.conv11(x)))
        x = F.relu(self.conv12_bn(self.conv12(x)))
        x = F.relu(self.conv21_bn(self.conv21(x)))
        x = F.relu(self.conv22_bn(self.conv22(x)))
        x = F.relu(self.conv31_bn(self.conv31(x)))
        x = F.relu(self.conv32_bn(self.conv32(x)))
        x = F.relu(self.conv41_bn(self.conv41(x)))
        # x = F.relu(self.conv2_bn(self.conv42(x))

        x = x.view(-1, 128 * 4 * 6 * 2)
        x = self.fc1_bn(self.fc1(x))
        # x = self.fc2(x)
        # x = self.fc3(x)

        return x
