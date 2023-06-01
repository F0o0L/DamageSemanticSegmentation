import torch
import torch.utils.model_zoo as modelZoo
from functools import partial
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Callable, List, Optional, Sequence, Tuple
from MyModeling.Backbone.MyMobileNetV2 import makeDivisible, ConvBNActivation

# a modification of torchvision implement of mobile net v3, which is used to be the backbone of deeplab v3+
modelUrls = {
    "MobileNetV3Large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
    "MobileNetV3Small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
}


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3, inplace=self.inplace) / 6


class SqueezeExcitation(nn.Module):
    def __init__(self,
                 inChannels: int,
                 squeezeFactor: int = 4) -> None:
        super().__init__()
        squeezeChannels = makeDivisible(inChannels // squeezeFactor, 8)
        self.fc1 = nn.Conv2d(inChannels, squeezeChannels, (1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeezeChannels, inChannels, (1, 1))

    def _scale(self,
               x: Tensor,
               inplace: bool) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        hardSigmoid = F.relu6(scale + 3, inplace=inplace) / 6
        return hardSigmoid

    def forward(self,
                x: Tensor) -> Tensor:
        scale = self._scale(x, True)
        return scale * x


class InvertedResidualConfig:
    def __init__(self,
                 inChannels: int,
                 kernel: int,
                 expandedChannels: int,
                 outChannels: int,
                 useSE: bool,
                 activation: str,
                 stride: int,
                 dilation: int,
                 widthMult: float):
        self.inputChannels = self.adjustChannels(inChannels, widthMult)
        self.kernel = kernel
        self.expandedChannels = self.adjustChannels(expandedChannels, widthMult)
        self.outChannels = self.adjustChannels(outChannels, widthMult)
        self.useSE = useSE
        self.useHS = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjustChannels(channels: int, widthMult: float):
        return makeDivisible(channels * widthMult, 8)


class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 normLayer: Callable[..., nn.Module],
                 SELayer: Callable[..., nn.Module] = SqueezeExcitation):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.useResConnect = cnf.stride == 1 and cnf.inputChannels == cnf.outChannels

        layers: List[nn.Module] = []
        activationLayer = Hswish if cnf.useHS else nn.ReLU

        # expand
        if cnf.expandedChannels != cnf.inputChannels:
            layers.append(ConvBNActivation(cnf.inputChannels, cnf.expandedChannels, kernelSize=1,
                                           normLayer=normLayer, activationLayer=activationLayer))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(ConvBNActivation(cnf.expandedChannels, cnf.expandedChannels, kernelSize=cnf.kernel,
                                       stride=stride, dilation=cnf.dilation, groups=cnf.expandedChannels,
                                       normLayer=normLayer, activationLayer=activationLayer))
        if cnf.useSE:
            layers.append(SELayer(cnf.expandedChannels))

        # project
        layers.append(ConvBNActivation(cnf.expandedChannels, cnf.outChannels, kernelSize=1, normLayer=normLayer,
                                       activationLayer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.outChannels = cnf.outChannels
        self._is_cn = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.useResConnect:
            result += x
        return result


class MobileNetV3(nn.Module):

    def __init__(
            self,
            arch: str,
            lowLevelOutputStride: int,
            invertedResidualSetting: List[InvertedResidualConfig],
            block: Optional[Callable[..., nn.Module]] = None,
            normLayer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        MobileNet V3 main class
        :param arch: MobileNetV3Small or MobileNetV3Large
        :param lowLevelOutputStride: low level output stride
        :param invertedResidualSetting: Network structure
        :param block: Module specifying inverted residual building block for mobilenet
        :param normLayer: Module specifying the normalization layer to use
        """
        super().__init__()
        self.arch = arch
        self.lowLevelOutputStride = lowLevelOutputStride

        if not invertedResidualSetting:
            raise ValueError("The inverted residual setting should not be empty")
        elif not (isinstance(invertedResidualSetting, Sequence) and
                  all([isinstance(s, InvertedResidualConfig) for s in invertedResidualSetting])):
            raise TypeError("The inverted residual setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if normLayer is None:
            normLayer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstConvOutputChannels = invertedResidualSetting[0].inputChannels
        layers.append(ConvBNActivation(3, firstConvOutputChannels, kernelSize=3, stride=2, normLayer=normLayer,
                                       activationLayer=Hswish))

        # building inverted residual blocks
        for cnf in invertedResidualSetting:
            layers.append(block(cnf, normLayer))

        # building last several layers
        lastConvInputChannels = invertedResidualSetting[-1].outChannels
        lastConvOutputChannels = 6 * lastConvInputChannels
        layers.append(ConvBNActivation(lastConvInputChannels, lastConvOutputChannels, kernelSize=1,
                                       normLayer=normLayer, activationLayer=Hswish))

        self.features = nn.Sequential(*layers)

    def _forwardImpl(self, x: Tensor) -> Tuple:
        lowLevelFeature = None
        if self.arch == 'MobileNetV3Large':
            if self.lowLevelOutputStride == 4:
                lowLevelFeature = self.features[0:3](x)
                x = self.features[3:](lowLevelFeature)
            elif self.lowLevelOutputStride == 8:
                lowLevelFeature = self.features[0:5](x)
                x = self.features[5:](lowLevelFeature)
            elif self.lowLevelOutputStride == 16:
                lowLevelFeature = self.features[0:8](x)
                x = self.features[8:](lowLevelFeature)
        elif self.arch == 'MobileNetV3Small':
            if self.lowLevelOutputStride == 4:
                lowLevelFeature = self.features[0:2](x)
                x = self.features[2:](lowLevelFeature)
            elif self.lowLevelOutputStride == 8:
                lowLevelFeature = self.features[0:3](x)
                x = self.features[3:](lowLevelFeature)
            elif self.lowLevelOutputStride == 16:
                lowLevelFeature = self.features[0:5](x)
                x = self.features[5:](lowLevelFeature)
        else:
            raise ValueError("arch should be one of 'MobileNetV3Large' and 'MobileNetV3Small'")
        return x, lowLevelFeature

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x, lowLevelFeature = self._forwardImpl(x)
        return x, lowLevelFeature


def _mobileNetV3Conf(arch: str,
                     outputStride: int,
                     widthMult: float = 1.0,
                     reducedTail: bool = False,
                     dilated: bool = False) -> list:
    reduceDivider = 2 if reducedTail else 1
    # dilation = 2 if dilated else 1
    currentStride = 2
    dilation = 1
    rate = 1

    bneckConf = partial(InvertedResidualConfig, widthMult=widthMult)
    if arch == "MobileNetV3Large":
        # inChannels,kernel,expandedChannels,outChannels,useSE,activation,stride
        setting = [[16, 3, 16, 16, False, "RE", 1],
                   [16, 3, 64, 24, False, "RE", 2],
                   [24, 3, 72, 24, False, "RE", 1],
                   [24, 5, 72, 40, True, "RE", 2],
                   [40, 5, 120, 40, True, "RE", 1],
                   [40, 5, 120, 40, True, "RE", 1],
                   [40, 3, 240, 80, False, "HS", 2],
                   [80, 3, 200, 80, False, "HS", 1],
                   [80, 3, 184, 80, False, "HS", 1],
                   [80, 3, 184, 80, False, "HS", 1],
                   [80, 3, 480, 112, True, "HS", 1],
                   [112, 3, 672, 112, True, "HS", 1],
                   [112, 5, 672, 160 // reduceDivider, True, "HS", 2],
                   [160 // reduceDivider, 5, 960 // reduceDivider, 160 // reduceDivider, True, "HS", 1],
                   [160 // reduceDivider, 5, 960 // reduceDivider, 160 // reduceDivider, True, "HS", 1]]
    elif arch == "MobileNetV3Small":
        setting = [[16, 3, 16, 16, True, "RE", 2],
                   [16, 3, 72, 24, False, "RE", 2],
                   [24, 3, 88, 24, False, "RE", 1],
                   [24, 5, 96, 40, True, "HS", 2],
                   [40, 5, 240, 40, True, "HS", 1],
                   [40, 5, 240, 40, True, "HS", 1],
                   [40, 5, 120, 48, True, "HS", 1],
                   [48, 5, 144, 48, True, "HS", 1],
                   [48, 5, 288, 96 // reduceDivider, True, "HS", 2],
                   [96 // reduceDivider, 5, 576 // reduceDivider, 96 // reduceDivider, True, "HS", 1],
                   [96 // reduceDivider, 5, 576 // reduceDivider, 96 // reduceDivider, True, "HS", 1]]
    else:
        raise ValueError("Unsupported model type {}".format(arch))
    invertedResidualSetting = []
    for inChannels, kernel, expandedChannels, outChannels, useSE, activation, s in setting:
        if outputStride == currentStride:
            stride = 1
            if dilated:
                dilation = rate
                rate *= s
        else:
            stride = s
            dilation = 1
            currentStride *= s
        invertedResidualSetting.append(
            bneckConf(inChannels, kernel, expandedChannels, outChannels, useSE, activation, stride, dilation))
    # invertedResidualSetting = [
    #     bneckConf(16, 3, 16, 16, False, "RE", 1, 1),
    #     bneckConf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
    #     bneckConf(24, 3, 72, 24, False, "RE", 1, 1),
    #     bneckConf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
    #     bneckConf(40, 5, 120, 40, True, "RE", 1, 1),
    #     bneckConf(40, 5, 120, 40, True, "RE", 1, 1),
    #     bneckConf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
    #     bneckConf(80, 3, 200, 80, False, "HS", 1, 1),
    #     bneckConf(80, 3, 184, 80, False, "HS", 1, 1),
    #     bneckConf(80, 3, 184, 80, False, "HS", 1, 1),
    #     bneckConf(80, 3, 480, 112, True, "HS", 1, 1),
    #     bneckConf(112, 3, 672, 112, True, "HS", 1, 1),
    #     bneckConf(112, 5, 672, 160 // reduceDivider, True, "HS", 2, dilation),  # C4
    #     bneckConf(160 // reduceDivider, 5, 960 // reduceDivider, 160 // reduceDivider, True, "HS", 1, dilation),
    #     bneckConf(160 // reduceDivider, 5, 960 // reduceDivider, 160 // reduceDivider, True, "HS", 1, dilation)]
    # elif arch == "MobileNetV3Small":
    #     # inChannels,kernel,expandedChannels,outChannels,useSE,activation,stride,dilation
    #     invertedResidualSetting = [
    #         bneckConf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
    #         bneckConf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
    #         bneckConf(24, 3, 88, 24, False, "RE", 1, 1),
    #         bneckConf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
    #         bneckConf(40, 5, 240, 40, True, "HS", 1, 1),
    #         bneckConf(40, 5, 240, 40, True, "HS", 1, 1),
    #         bneckConf(40, 5, 120, 48, True, "HS", 1, 1),
    #         bneckConf(48, 5, 144, 48, True, "HS", 1, 1),
    #         bneckConf(48, 5, 288, 96 // reduceDivider, True, "HS", 2, dilation),  # C4
    #         bneckConf(96 // reduceDivider, 5, 576 // reduceDivider, 96 // reduceDivider, True, "HS", 1, dilation),
    #         bneckConf(96 // reduceDivider, 5, 576 // reduceDivider, 96 // reduceDivider, True, "HS", 1, dilation)]
    # else:
    #     raise ValueError("Unsupported model type {}".format(arch))

    return invertedResidualSetting


def _mobileNetV3Model(
        arch: str,
        lowLevelOutputStride: int,
        invertedResidualSetting: List[InvertedResidualConfig],
        preTrained: bool,
        progress: bool):
    model = MobileNetV3(arch=arch,
                        lowLevelOutputStride=lowLevelOutputStride,
                        invertedResidualSetting=invertedResidualSetting)
    if preTrained:
        if modelUrls.get(arch, None) is None:
            raise ValueError("No checkpoint is available for model type {}".format(arch))
        stateDict = modelZoo.load_url(url=modelUrls[arch],
                                      model_dir='MyModeling/Backbone/Weights',
                                      # model_dir='Backbone/Weights',
                                      # model_dir='Weights',
                                      progress=progress,
                                      map_location='cpu')
        model.load_state_dict(stateDict, strict=False)
    return model


def mobileNetV3Large(outputStride: int,
                     lowLevelOutputStride: int,
                     preTrained: bool = False,
                     dilated: bool = False,
                     progress: bool = True) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture
    :param outputStride:output stride
    :param lowLevelOutputStride: low level output stride
    :param preTrained: If True, returns a model pre-trained on ImageNet
    :param dilated: whether to set dilation==2 in some layers
    :param progress: If True, displays a progress bar of the download to stderr
    """
    assert lowLevelOutputStride in [4, 8, 16]
    assert outputStride in [8, 16, 32]
    assert lowLevelOutputStride < outputStride
    arch = "MobileNetV3Large"
    invertedResidualSetting = _mobileNetV3Conf(arch, outputStride=outputStride, dilated=dilated)
    return _mobileNetV3Model(arch=arch,
                             lowLevelOutputStride=lowLevelOutputStride,
                             invertedResidualSetting=invertedResidualSetting,
                             preTrained=preTrained,
                             progress=progress)


def mobileNetV3Small(outputStride: int,
                     lowLevelOutputStride: int,
                     preTrained: bool = False,
                     dilated: bool = False,
                     progress: bool = True) -> MobileNetV3:
    """
    Constructs a small MobileNetV3 architecture
    :param outputStride:output stride
    :param lowLevelOutputStride: low level output stride
    :param preTrained: If True, returns a model pre-trained on ImageNet
    :param dilated: whether to set dilation==2 in some layers
    :param progress: If True, displays a progress bar of the download to stderr
    """
    assert lowLevelOutputStride in [4, 8, 16]
    assert outputStride in [8, 16, 32]
    assert lowLevelOutputStride < outputStride
    arch = "MobileNetV3Small"
    invertedResidualSetting = _mobileNetV3Conf(arch, outputStride=outputStride, dilated=dilated)
    return _mobileNetV3Model(arch=arch,
                             lowLevelOutputStride=lowLevelOutputStride,
                             invertedResidualSetting=invertedResidualSetting,
                             preTrained=preTrained,
                             progress=progress)


if __name__ == '__main__':
    # modelSmall = mobileNetV3Small(outputStride=8, lowLevelOutputStride=4, preTrained=False)
    # res1 = torch.rand(1, 3, 192, 256)
    # res1, lowLevel1 = modelSmall(res1)

    modelLarge = mobileNetV3Large(outputStride=8, lowLevelOutputStride=4, preTrained=False,dilated=True)
    res2 = torch.rand(1, 3, 512, 512)
    res2, lowLevel2 = modelLarge(res2)

    # print(res1.size())
    # print(lowLevel1.size())
    print(res2.size())
    print(lowLevel2.size())
    print(modelLarge)
