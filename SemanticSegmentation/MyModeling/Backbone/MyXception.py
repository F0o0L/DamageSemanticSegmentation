import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelZoo
from torch import Tensor
from typing import Tuple


class AlignedXception(nn.Module):
    def __init__(self, preTrained: bool, outputStride=8) -> None:
        super(AlignedXception, self).__init__()

        assert outputStride in [8, 16]
        if outputStride == 8:
            entryBlock3Stride = 1
        else:
            entryBlock3Stride = 2
        middleBlockDilation = 1
        exitBlockDilations = (1, 2)

        # Entry Flow
        self.conv1 = nn.Conv2d(3, 32, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, startWithRelu=False)
        self.block2 = Block(128, 256, 2, startWithRelu=False)
        self.block3 = Block(256, 728, entryBlock3Stride, startWithRelu=True)

        # Middle flow
        self.block4 = Block(728, 728, dilation=middleBlockDilation)
        self.block5 = Block(728, 728, dilation=middleBlockDilation)
        self.block6 = Block(728, 728, dilation=middleBlockDilation)
        self.block7 = Block(728, 728, dilation=middleBlockDilation)
        self.block8 = Block(728, 728, dilation=middleBlockDilation)
        self.block9 = Block(728, 728, dilation=middleBlockDilation)
        self.block10 = Block(728, 728, dilation=middleBlockDilation)
        self.block11 = Block(728, 728, dilation=middleBlockDilation)
        self.block12 = Block(728, 728, dilation=middleBlockDilation)
        self.block13 = Block(728, 728, dilation=middleBlockDilation)
        self.block14 = Block(728, 728, dilation=middleBlockDilation)
        self.block15 = Block(728, 728, dilation=middleBlockDilation)
        self.block16 = Block(728, 728, dilation=middleBlockDilation)
        self.block17 = Block(728, 728, dilation=middleBlockDilation)
        self.block18 = Block(728, 728, dilation=middleBlockDilation)
        self.block19 = Block(728, 728, dilation=middleBlockDilation)

        # Exit flow
        self.block20 = Block1(728, 1024, dilation=exitBlockDilations[0])
        self.conv3 = SeparableConv2d(1024, 1536, dilation=exitBlockDilations[1])
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 1536, dilation=exitBlockDilations[1])
        self.bn4 = nn.BatchNorm2d(1536)
        self.conv5 = SeparableConv2d(1536, 2048, dilation=exitBlockDilations[1])
        self.bn5 = nn.BatchNorm2d(2048)

        if preTrained:
            self.preTrain()

    def forward(self, x: Tensor) -> Tuple:  # 3,513,513
        # Entry Flow
        x = self.conv1(x)  # 32,257,257
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)  # 64,257,257
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)  # 128,129,129
        # add relu here
        x = self.relu(x)
        lowLevelFeat = x  # 128,129,129

        x = self.block2(x)  # 256,65,65
        x = self.block3(x)  # 728,33,33

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)  # 1024,33,33
        x = self.relu(x)
        x = self.conv3(x)  # 1536,33,33
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)  # 1536,33,33
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)  # 2048,33,33
        x = self.bn5(x)
        x = self.relu(x)

        return x, lowLevelFeat

    def preTrain(self):
        pretrainDict = modelZoo.load_url(url='http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
                                         model_dir='MyModeling/Backbone/Weights',
                                         # model_dir='Backbone/Weights',
                                         # model_dir='Weights',
                                         )
        stateDicts = self.state_dict()

        for k, v in pretrainDict.items():
            if k in stateDicts:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block11'):
                    stateDicts[k] = v
                    stateDicts[k.replace('block11', 'block12')] = v
                    stateDicts[k.replace('block11', 'block13')] = v
                    stateDicts[k.replace('block11', 'block14')] = v
                    stateDicts[k.replace('block11', 'block15')] = v
                    stateDicts[k.replace('block11', 'block16')] = v
                    stateDicts[k.replace('block11', 'block17')] = v
                    stateDicts[k.replace('block11', 'block18')] = v
                    stateDicts[k.replace('block11', 'block19')] = v
                elif k.startswith('block12'):
                    stateDicts[k.replace('block12', 'block20')] = v
                elif k.startswith('bn3'):
                    stateDicts[k] = v
                    stateDicts[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    stateDicts[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    stateDicts[k.replace('bn4', 'bn5')] = v
                else:
                    stateDicts[k] = v
        self.load_state_dict(state_dict=stateDicts, strict=False)


def fixedPadding(x: Tensor,
                 kernelSize: int,
                 dilation: int) -> Tensor:
    pad = dilation * (kernelSize - 1)
    padBeg = pad // 2
    padEnd = pad - padBeg
    output = F.pad(x, [padBeg, padEnd, padBeg, padEnd])
    return output


class SeparableConv2d(nn.Module):
    def __init__(self,
                 inChannel: int,
                 outChannel: int,
                 kernelSize: int = 3,
                 stride: int = 1,
                 dilation: int = 1,
                 bias: bool = False) -> None:
        super(SeparableConv2d, self).__init__()
        self.kernelSize = kernelSize
        self.stride = stride
        self.dilation = dilation
        self.conv1 = nn.Conv2d(inChannel, inChannel, (kernelSize, kernelSize), (stride, stride), (0, 0),
                               (dilation, dilation), groups=inChannel, bias=bias)
        self.bn = nn.BatchNorm2d(inChannel)
        self.pointwise = nn.Conv2d(inChannel, outChannel, kernel_size=(1, 1), bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = fixedPadding(x, self.kernelSize, self.dilation)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 inChannel: int,
                 outChannel: int,
                 stride: int = 1,
                 dilation: int = 1,
                 startWithRelu: bool = True) -> None:
        super(Block, self).__init__()
        assert stride == 1 or stride == 2

        rep = []
        self.relu = nn.ReLU(inplace=True)
        rep.append(self.relu)
        rep.append(SeparableConv2d(inChannel, outChannel, 3, 1, dilation=dilation))
        rep.append(nn.BatchNorm2d(outChannel))

        rep.append(self.relu)
        rep.append(SeparableConv2d(outChannel, outChannel, 3, 1, dilation=dilation))
        rep.append(nn.BatchNorm2d(outChannel))

        rep.append(self.relu)
        rep.append(SeparableConv2d(outChannel, outChannel, 3, stride=stride, dilation=dilation))
        rep.append(nn.BatchNorm2d(outChannel))

        if inChannel != outChannel or stride != 1:
            self.skip = nn.Conv2d(inChannel, outChannel, (1, 1), stride=(stride, stride), bias=False)
            self.skipbn = nn.BatchNorm2d(outChannel)
        else:
            self.skip = None
        if not startWithRelu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp: Tensor) -> Tensor:
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip

        return x


class Block1(nn.Module):
    def __init__(self,
                 inChannel: int,
                 outChannel: int,
                 stride: int = 1,
                 dilation: int = 1,
                 startWithRelu: bool = True) -> None:
        super(Block1, self).__init__()
        assert stride == 1 or stride == 2

        rep = []
        self.relu = nn.ReLU(inplace=True)
        rep.append(self.relu)
        rep.append(SeparableConv2d(inChannel, inChannel, 3, 1, dilation=dilation))
        rep.append(nn.BatchNorm2d(inChannel))

        rep.append(self.relu)
        rep.append(SeparableConv2d(inChannel, outChannel, 3, 1, dilation=dilation))
        rep.append(nn.BatchNorm2d(outChannel))

        rep.append(self.relu)
        rep.append(SeparableConv2d(outChannel, outChannel, 3, stride=stride, dilation=dilation))
        rep.append(nn.BatchNorm2d(outChannel))

        if inChannel != outChannel or stride != 1:
            self.skip = nn.Conv2d(inChannel, outChannel, (1, 1), stride=(stride, stride), bias=False)
            self.skipbn = nn.BatchNorm2d(outChannel)
        else:
            self.skip = None
        if not startWithRelu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp: Tensor) -> Tensor:
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip

        return x


if __name__ == '__main__':
    import torch

    ax = AlignedXception(preTrained=False, outputStride=8)
    ip = torch.rand(2, 3, 512, 512)
    output, low = ax(ip)
    print(output.size())
    print(low.size())
