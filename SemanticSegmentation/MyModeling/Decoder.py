import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Decoder(nn.Module):
    def __init__(self,
                 numClasses: int,
                 lowLevelInChannel: int) -> None:
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(lowLevelInChannel, 48, (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        self.lastConv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv2d(256, numClasses, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self,
                lowLevelFeature: Tensor,
                x: Tensor) -> Tensor:  # lowLevelFeat:128,129,129 x:256,33,33
        lowLevelFeature = self.conv1(lowLevelFeature)  # 48,129,129
        lowLevelFeature = self.bn1(lowLevelFeature)
        lowLevelFeature = self.relu(lowLevelFeature)

        x = F.interpolate(x, size=lowLevelFeature.size()[2:], mode='bilinear', align_corners=True)  # 256,129,129
        x = torch.cat((x, lowLevelFeature), dim=1)  # 304,129,129
        x = self.lastConv(x)  # 12,129,129
        return x


class Decoder1(nn.Module):
    def __init__(self,
                 numClasses: int,
                 lowLevelInChannel: int) -> None:
        super(Decoder1, self).__init__()
        self.conv1 = nn.Conv2d(128, numClasses, (1, 1), bias=False)
        self.branch3 = nn.Conv2d(lowLevelInChannel, numClasses, (1, 1), bias=False)

    def forward(self,
                lowLevelFeature: Tensor,
                x: Tensor) -> Tensor:
        x = nn.functional.interpolate(x, lowLevelFeature.size()[2:])
        x = self.conv1(x)
        lowLevelFeat = self.branch3(lowLevelFeature)
        return x + lowLevelFeat


def buildDecoder(numClasses: int,
                 lowLevelInChannel: int,
                 mode: str) -> nn.Module:
    if mode == 'LRASPP':
        return Decoder1(numClasses, lowLevelInChannel)
    elif mode == 'ASPP':
        return Decoder(numClasses, lowLevelInChannel)
    else:
        raise ValueError("mode should be one of 'ASPP' or 'LRASPP'")


if __name__ == '__main__':
    a = torch.rand(1, 128, 7, 7)
    b = torch.rand(1, 48, 14, 14)
    d = Decoder1(12, 48)
    d(a, b)
