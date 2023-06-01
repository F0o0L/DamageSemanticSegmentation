import torch.nn as nn
from torch.nn import functional as F


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3, inplace=self.inplace) / 6


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3, inplace=self.inplace) / 6


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgPool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MobileBottleneck(nn.Module):
    def __init__(self, inChannels, outChannels, latentChannels, kernel, stride, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.useResConnect = stride == 1 and inChannels == outChannels

        if nl == 'RE':
            nonlinear_layer = nn.ReLU  # or ReLU6
        elif nl == 'HS':
            nonlinear_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inChannels, latentChannels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(latentChannels),
            nonlinear_layer(inplace=True),
            # dw
            nn.Conv2d(latentChannels, latentChannels, kernel, stride, padding, groups=latentChannels, bias=False),
            nn.BatchNorm2d(latentChannels),
            SELayer(latentChannels),
            nonlinear_layer(inplace=True),
            # pw-linear
            nn.Conv2d(latentChannels, outChannels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outChannels)
        )

    def forward(self, x):
        if self.useResConnect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3Large(nn.Module):
    def __init__(self):
        super(MobileNetV3Large, self).__init__()
        # Start Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            Hswish(inplace=True)
        )
        # MobileBottleneck blocks
        self.conv2 = MobileBottleneck(16, 16, 16, 3, 1, False, 'RE')
        self.conv3 = MobileBottleneck(16, 24, 64, 3, 2, False, 'RE')
        self.conv4 = MobileBottleneck(24, 24, 72, 3, 1, False, 'RE')
        self.conv5 = MobileBottleneck(24, 40, 72, 5, 2, True, 'RE')
        self.conv6 = MobileBottleneck(40, 40, 120, 5, 1, True, 'RE')
        self.conv7 = MobileBottleneck(40, 40, 120, 5, 1, True, 'RE')
        self.conv8 = MobileBottleneck(40, 80, 240, 3, 2, False, 'HS')
        self.conv9 = MobileBottleneck(80, 80, 200, 3, 1, False, 'HS')
        self.conv10 = MobileBottleneck(80, 80, 184, 3, 1, False, 'HS')
        self.conv11 = MobileBottleneck(80, 80, 184, 3, 1, False, 'HS')
        self.conv12 = MobileBottleneck(80, 112, 480, 3, 1, True, 'HS')
        self.conv13 = MobileBottleneck(112, 112, 672, 3, 1, True, 'HS')
        self.conv14 = MobileBottleneck(112, 160, 672, 5, 2, True, 'HS')
        self.conv15 = MobileBottleneck(160, 160, 960, 5, 1, True, 'HS')
        self.conv16 = MobileBottleneck(160, 160, 960, 5, 1, True, 'HS')
        # Last Conv
        self.conv17 = nn.Sequential(
            nn.Conv2d(160, 960, 1, 1, 0, bias=False),
            nn.BatchNorm2d(960),
            Hswish(inplace=True)
        )

    def forward(self, x):
        # feature extraction
        x = self.conv1(x)  # out: B * 16 * 112 * 112
        x = self.conv2(x)  # out: B * 16 * 112 * 112
        x = self.conv3(x)  # out: B * 24 * 56 * 56
        x = self.conv4(x)  # out: B * 24 * 56 * 56
        x = self.conv5(x)  # out: B * 40 * 28 * 28
        x = self.conv6(x)  # out: B * 40 * 28 * 28
        x = self.conv7(x)  # out: B * 40 * 28 * 28
        x = self.conv8(x)  # out: B * 80 * 14 * 14
        x = self.conv9(x)  # out: B * 80 * 14 * 14
        x = self.conv10(x)  # out: B * 80 * 14 * 14
        x = self.conv11(x)  # out: B * 80 * 14 * 14
        x = self.conv12(x)  # out: B * 112 * 14 * 14
        x = self.conv13(x)  # out: B * 112 * 14 * 14
        lowLevelFeature = x
        x = self.conv14(x)  # out: B * 160 * 7 * 7
        x = self.conv15(x)  # out: B * 160 * 7 * 7
        x = self.conv16(x)  # out: B * 160 * 7 * 7
        x = self.conv17(x)  # out: B * 960 * 7 * 7
        return x, lowLevelFeature


class MobileNetV3Small(nn.Module):
    def __init__(self):
        super(MobileNetV3Small, self).__init__()
        # Start Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            Hswish(inplace=True)
        )
        # MobileBottleneck blocks
        self.conv2 = MobileBottleneck(16, 16, 16, 3, 2, True, 'RE')
        self.conv3 = MobileBottleneck(16, 24, 72, 3, 2, False, 'RE')
        self.conv4 = MobileBottleneck(24, 24, 88, 3, 1, False, 'RE')
        self.conv5 = MobileBottleneck(24, 40, 96, 5, 2, True, 'HS')
        self.conv6 = MobileBottleneck(40, 40, 240, 5, 1, True, 'HS')
        self.conv7 = MobileBottleneck(40, 40, 240, 5, 1, True, 'HS')
        self.conv8 = MobileBottleneck(40, 48, 120, 5, 1, True, 'HS')
        self.conv9 = MobileBottleneck(48, 48, 144, 5, 1, True, 'HS')
        self.conv10 = MobileBottleneck(48, 96, 288, 5, 2, True, 'HS')
        self.conv11 = MobileBottleneck(96, 96, 576, 3, 1, True, 'HS')
        self.conv12 = MobileBottleneck(96, 96, 576, 3, 1, True, 'HS')
        # Last Conv
        self.conv13 = nn.Sequential(
            nn.Conv2d(96, 576, 1, 1, 0, bias=False),
            nn.BatchNorm2d(576),
            Hswish(inplace=True)
        )

    def forward(self, x):
        # feature extraction
        x = self.conv1(x)  # out: B * 16 * 112 * 112
        x = self.conv2(x)  # out: B * 16 * 56 * 56
        x = self.conv3(x)  # out: B * 24 * 28 * 28
        x = self.conv4(x)  # out: B * 24 * 28 * 28
        x = self.conv5(x)  # out: B * 40 * 14 * 14
        x = self.conv6(x)  # out: B * 40 * 14 * 14
        x = self.conv7(x)  # out: B * 40 * 14 * 14
        x = self.conv8(x)  # out: B * 48 * 14 * 14
        x = self.conv9(x)  # out: B * 48 * 14 * 14
        lowLevelFeature = x
        x = self.conv10(x)  # out: B * 96 * 7 * 7
        x = self.conv11(x)  # out: B * 96 * 7 * 7
        x = self.conv12(x)  # out: B * 96 * 7 * 7
        x = self.conv13(x)  # out: B * 576 * 7 * 7
        return x, lowLevelFeature
