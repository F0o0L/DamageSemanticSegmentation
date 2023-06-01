import torch
import torch.utils.model_zoo as modelZoo
from torch import nn
from torch import Tensor
from typing import Callable, Optional, List, Tuple


def makeDivisible(inputChannels: float,
                  divisor: int,
                  minValue: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    """
    if minValue is None:
        minValue = divisor
    outputChannels = max(minValue, int(inputChannels + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if outputChannels < 0.9 * inputChannels:
        outputChannels += divisor
    return outputChannels


class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 inChannels: int,
                 outChannels: int,
                 kernelSize: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 normLayer: Optional[Callable[..., nn.Module]] = None,
                 activationLayer: Optional[Callable[..., nn.Module]] = None,
                 dilation: int = 1) -> None:
        padding = (kernelSize - 1) // 2 * dilation
        if normLayer is None:
            normLayer = nn.BatchNorm2d
        if activationLayer is None:
            activationLayer = nn.ReLU6
        super().__init__(
            nn.Conv2d(inChannels, outChannels, (kernelSize, kernelSize), (stride, stride), (padding, padding),
                      (dilation, dilation), groups=groups, bias=False),
            normLayer(outChannels),
            activationLayer(inplace=True)
        )
        self.outChannels = outChannels


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class InvertedResidual(nn.Module):
    def __init__(
            self,
            inChannels: int,
            outChannels: int,
            stride: int,
            expandRatio: int,
            dilation: Optional[int] = 1,
            normLayer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if normLayer is None:
            normLayer = nn.BatchNorm2d

        hiddenDim = int(round(inChannels * expandRatio))
        self.useResConnect = self.stride == 1 and inChannels == outChannels

        layers: List[nn.Module] = []
        if expandRatio != 1:
            # pw
            layers.append(ConvBNReLU(inChannels, hiddenDim, kernelSize=1, normLayer=normLayer))
        layers.extend([
            # dw
            ConvBNReLU(hiddenDim, hiddenDim, stride=stride, groups=hiddenDim, normLayer=normLayer, dilation=dilation),
            # pw-linear
            nn.Conv2d(hiddenDim, outChannels, (1, 1), (1, 1), (0, 0), bias=False),
            normLayer(outChannels)])
        self.conv = nn.Sequential(*layers)
        self.outChannels = outChannels
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.useResConnect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
            self,
            outputStride: int,
            lowLevelOutputStride: int,
            widthMult: float = 1.0,
            invertedResidualSetting: Optional[List[List[int]]] = None,
            roundNearest: int = 8,
            block: Optional[Callable[..., nn.Module]] = None,
            normLayer: Optional[Callable[..., nn.Module]] = None,
            dilated: bool = False) -> None:
        """
        MobileNet V2 main class
        :param outputStride:output stride
        :param lowLevelOutputStride: low level output stride
        :param widthMult: Width multiplier - adjusts number of channels in each layer by this amount
        :param invertedResidualSetting: Network structure
        :param roundNearest: Round the number of channels in each layer to be a multiple of this number.
                            Set to 1 to turn off rounding
        :param block: Module specifying inverted residual building block for mobilenet
        :param normLayer: Module specifying the normalization layer to use
        :param dilated: whether to set dilation==2 in some layer
        """
        super(MobileNetV2, self).__init__()
        # dilation = 2 if dilated else 1
        assert lowLevelOutputStride in [4, 8, 16]
        assert outputStride in [8, 16, 32]
        assert outputStride > lowLevelOutputStride
        self.lowLevelOutputStride = lowLevelOutputStride
        self.outputStride = outputStride
        currentStride = 1
        dilation = 1
        rate = 1

        if block is None:
            block = InvertedResidual

        if normLayer is None:
            normLayer = nn.BatchNorm2d

        inputChannel = 32
        lastChannel = 1280

        if invertedResidualSetting is None:
            invertedResidualSetting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(invertedResidualSetting) == 0 or len(invertedResidualSetting[0]) != 4:
            raise ValueError("inverted residual setting should be non-empty and a 4-element list, got {}".format(
                invertedResidualSetting))

        # building first layer
        inputChannel = makeDivisible(inputChannel * widthMult, roundNearest)
        lastChannel = makeDivisible(lastChannel * max(1.0, widthMult), roundNearest)
        features: List[nn.Module] = [ConvBNReLU(3, inputChannel, stride=2, normLayer=normLayer)]
        currentStride *= 2
        # building inverted residual blocks
        for t, c, n, s in invertedResidualSetting:
            outputChannel = makeDivisible(c * widthMult, roundNearest)
            if currentStride == self.outputStride:
                ss = 1
                if dilated:
                    dilation = rate
                    rate *= s
            else:
                ss = s
                dilation = 1
                currentStride *= s
            for i in range(n):
                stride = ss if i == 0 else 1
                features.append(block(inChannels=inputChannel,
                                      outChannels=outputChannel,
                                      stride=stride,
                                      expandRatio=t,
                                      normLayer=normLayer,
                                      dilation=dilation))
                inputChannel = outputChannel
        features.append(ConvBNReLU(inChannels=inputChannel,
                                   outChannels=lastChannel,
                                   kernelSize=1,
                                   normLayer=normLayer,
                                   dilation=dilation))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

    def _forwardImpl(self, x: Tensor) -> Tuple:
        lowLevelFeature = None
        if self.lowLevelOutputStride == 4:
            lowLevelFeature = self.features[0:3](x)
            x = self.features[3:](lowLevelFeature)
        elif self.lowLevelOutputStride == 8:
            lowLevelFeature = self.features[0:5](x)
            x = self.features[5:](lowLevelFeature)
        elif self.lowLevelOutputStride == 16:
            lowLevelFeature = self.features[0:8](x)
            x = self.features[8:](lowLevelFeature)
        return x, lowLevelFeature

    def forward(self, x: Tensor) -> Tuple:
        return self._forwardImpl(x)


def mobileNetV2(outputStride: int,
                lowLevelOutputStride: int,
                preTrained: bool = False,
                dilated: bool = False) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture
    :param outputStride:
    :param lowLevelOutputStride: low level output stride
    :param preTrained: If True, returns a model pre-trained on ImageNet
    :param dilated: whether to set dilation==2 in some layer
    """
    model = MobileNetV2(outputStride=outputStride, lowLevelOutputStride=lowLevelOutputStride, dilated=dilated)
    if preTrained:
        stateDict = modelZoo.load_url(url='https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
                                      model_dir='MyModeling/Backbone/Weights',
                                      # model_dir='Backbone/Weights',
                                      # model_dir='Weights',
                                      map_location='cpu')
        model.load_state_dict(stateDict, strict=False)
    return model


if __name__ == '__main__':
    mv2 = mobileNetV2(outputStride=8, lowLevelOutputStride=4, preTrained=True, dilated=True)
    res = torch.rand(1, 3, 192, 256)
    res, low = mv2(res)
    print(res.size())
    print(low.size())
    print(mv2)
