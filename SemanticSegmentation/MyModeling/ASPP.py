import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ASPP(nn.Module):
    def __init__(self,
                 inChannel: int,
                 dilations: list,
                 modify: bool = False) -> None:
        super(ASPP, self).__init__()
        self.modify = modify
        dilations = dilations
        # dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inChannel, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inChannel, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inChannel, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inChannel, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.globalAveragePool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inChannel, 256, (1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        if self.modify:
            self.conv1 = nn.Conv2d(1024, 256, (1, 1), bias=False)
        else:
            self.conv1 = nn.Conv2d(1280, 256, (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.aspp1(x)  # 256,33,33
        x2 = self.aspp2(x)  # 256,33,33
        x3 = self.aspp3(x)  # 256,33,33
        x4 = self.aspp4(x)  # 256,33,33
        if not self.modify:
            x5 = self.globalAveragePool(x)  # 256,1,1
            x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # 256,33,33
            x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # 1280,33,33
        else:
            x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x


class _ASPPModule(nn.Module):
    def __init__(self,
                 inChannel: int,
                 outChannel: int,
                 kernelSize: int,
                 padding: int,
                 dilation: int) -> None:
        super(_ASPPModule, self).__init__()
        self.atrousConv = nn.Conv2d(inChannel, outChannel, kernel_size=(kernelSize, kernelSize), stride=(1, 1),
                                    padding=(padding, padding), dilation=(dilation, dilation), bias=False)
        self.batchNorm = nn.BatchNorm2d(outChannel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.atrousConv(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        return x


class LiteRASPP(nn.Module):
    def __init__(self, inChannels: int) -> None:
        super(LiteRASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(inChannels, 128, (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            # nn.AvgPool2d(49, stride=(16, 20), padding=24),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inChannels, 128, (1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        branch1 = self.branch1(x)
        # branch2 = nn.functional.interpolate(self.branch2(input), size=branch1.size()[2:])
        branch2 = self.branch2(x)
        return branch1 * branch2


def buildASPP(inChannel: int,
              mode: str,
              dilations: list,
              modify: bool = None) -> nn.Module:
    if mode == 'ASPP':
        aspp = ASPP(inChannel,
                    dilations=dilations,
                    modify=modify)
        return aspp
    elif mode == 'LRASPP':
        lraspp = LiteRASPP(inChannel)
        return lraspp
    else:
        raise ValueError("mode should be one of 'ASPP' and 'LRASPP'")


if __name__ == '__main__':
    a = torch.rand(1, 3, 7, 7)
    la = LiteRASPP(3)
    la(a)
