import torch.nn as nn
import torch.utils.model_zoo as modelZoo

webroot = 'http://dl.yf.io/drn/'

modelUrls = {
    'resnet50': "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    'drn-c-26': webroot + 'drn_c_26-ddedf421.pth',
    'drn-c-42': webroot + 'drn_c_42-9d336e8c.pth',
    'drn-c-58': webroot + 'drn_c_58-0a53a92c.pth',
    'drn-d-22': webroot + 'drn_d_22-4bd2f8ea.pth',
    'drn-d-38': webroot + 'drn_d_38-eebb45f0.pth',
    'drn-d-54': webroot + 'drn_d_54-0e0534ff.pth',
    'drn-d-105': webroot + 'drn_d_105-12b40979.pth'
}
# modelDir = 'Weights'
# modelDir = 'Backbone/Weights'
modelDir = 'MyModeling/Backbone/Weights'


def conv3x3(inChannels, outChannels, stride=(1, 1), padding=(1, 1), dilation=(1, 1)):
    return nn.Conv2d(inChannels, outChannels, kernel_size=(3, 3), stride=stride, padding=padding, bias=False,
                     dilation=dilation)


class BasicBlock(nn.Module):
    def __init__(self, inChannels, channels, stride=1, downsample=None, dilation=(1, 1), residual=True,
                 BatchNorm=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inChannels, channels, stride, padding=dilation[0], dilation=dilation[0])
        self.bn1 = BatchNorm(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels, padding=dilation[1], dilation=dilation[1])
        self.bn2 = BatchNorm(channels)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inChannels, channels, stride=1, downsample=None, dilation=(1, 1), residual=True,
                 BatchNorm=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, channels, kernel_size=(1, 1), bias=False)
        self.bn1 = BatchNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(stride, stride),
                               padding=(dilation[1], dilation[1]), bias=False, dilation=(dilation[1], dilation[1]))
        self.bn2 = BatchNorm(channels)
        self.conv3 = nn.Conv2d(channels, channels * self.expansion, kernel_size=(1, 1), bias=False)
        self.bn3 = BatchNorm(channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class DRNA(nn.Module):

    def __init__(self, block, layers, BatchNorm=nn.BatchNorm2d):
        super(DRNA, self).__init__()
        self.inChannels = 64
        self.outDim = 512 * block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._makeLayer(block, 64, layers[0], BatchNorm=BatchNorm)
        self.layer2 = self._makeLayer(block, 128, layers[1], stride=2, BatchNorm=BatchNorm)
        self.layer3 = self._makeLayer(block, 256, layers[2], stride=1, dilation=2, BatchNorm=BatchNorm)
        self.layer4 = self._makeLayer(block, 512, layers[3], stride=1, dilation=4, BatchNorm=BatchNorm)

    def _makeLayer(self, block, channels, blocks, stride=1, dilation=1, BatchNorm=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inChannels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inChannels, channels * block.expansion, kernel_size=(1, 1), stride=(stride, stride),
                          bias=False),
                BatchNorm(channels * block.expansion),
            )

        layers = [block(self.inChannels, channels, stride, downsample, BatchNorm=BatchNorm)]
        self.inChannels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inChannels, channels, dilation=(dilation, dilation), BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxPool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class DRN(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 arch='D',
                 channels=(16, 32, 64, 128, 256, 512, 512, 512),
                 outputStride=8,
                 BatchNorm=nn.BatchNorm2d):
        super(DRN, self).__init__()
        assert outputStride in [8, 16]
        if outputStride == 8:
            layer5Stride = 1
        else:
            layer5Stride = 2
        self.inChannels = channels[0]
        self.outDim = channels[-1]
        self.arch = arch

        if arch == 'C':
            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
            self.bn1 = BatchNorm(channels[0])
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._makeLayer(BasicBlock, channels[0], layers[0], stride=1, BatchNorm=BatchNorm)
            self.layer2 = self._makeLayer(BasicBlock, channels[1], layers[1], stride=2, BatchNorm=BatchNorm)

        elif arch == 'D':
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, channels[0], kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False),
                BatchNorm(channels[0]),
                nn.ReLU(inplace=True)
            )

            self.layer1 = self._makeConvLayers(channels[0], layers[0], stride=1, BatchNorm=BatchNorm)
            self.layer2 = self._makeConvLayers(channels[1], layers[1], stride=2, BatchNorm=BatchNorm)

        self.layer3 = self._makeLayer(block, channels[2], layers[2], stride=2, BatchNorm=BatchNorm)
        self.layer4 = self._makeLayer(block, channels[3], layers[3], stride=2, BatchNorm=BatchNorm)
        self.layer5 = self._makeLayer(block, channels[4], layers[4], stride=layer5Stride, dilation=2, newLevel=False,
                                      BatchNorm=BatchNorm)
        self.layer6 = None if layers[5] == 0 else self._makeLayer(block, channels[5], layers[5], dilation=4,
                                                                  newLevel=False, BatchNorm=BatchNorm)

        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else self._makeLayer(BasicBlock, channels[6], layers[6], dilation=2,
                                                                      newLevel=False, residual=False,
                                                                      BatchNorm=BatchNorm)
            self.layer8 = None if layers[7] == 0 else self._makeLayer(BasicBlock, channels[7], layers[7], dilation=1,
                                                                      newLevel=False, residual=False,
                                                                      BatchNorm=BatchNorm)
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else self._makeConvLayers(channels[6], layers[6], dilation=2,
                                                                           BatchNorm=BatchNorm)
            self.layer8 = None if layers[7] == 0 else self._makeConvLayers(channels[7], layers[7], dilation=1,
                                                                           BatchNorm=BatchNorm)

    def _makeLayer(self, block, channels, blocks, stride=1, dilation=1, newLevel=True, residual=True,
                   BatchNorm=nn.BatchNorm2d):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inChannels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inChannels, channels * block.expansion, kernel_size=(1, 1), stride=(stride, stride),
                          bias=False),
                BatchNorm(channels * block.expansion),
            )
        layers = [block(self.inChannels, channels, stride, downsample,
                        dilation=(1, 1) if dilation == 1 else (dilation // 2 if newLevel else (dilation, dilation)),
                        residual=residual, BatchNorm=BatchNorm)]
        self.inChannels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inChannels, channels, residual=residual, dilation=(dilation, dilation), BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _makeConvLayers(self, channels, convs, stride=1, dilation=1, BatchNorm=None):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.inChannels, channels,
                          kernel_size=(3, 3),
                          stride=(stride, stride) if i == 0 else (1, 1),
                          padding=(dilation, dilation),
                          bias=False,
                          dilation=(dilation, dilation)),
                BatchNorm(channels),
                nn.ReLU(inplace=True)])
            self.inChannels = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D':
            x = self.layer0(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        lowLevelFeat = x

        x = self.layer4(x)
        x = self.layer5(x)

        if self.layer6 is not None:
            x = self.layer6(x)

        if self.layer7 is not None:
            x = self.layer7(x)

        if self.layer8 is not None:
            x = self.layer8(x)

        return x, lowLevelFeat


def DRNA50(BatchNorm=nn.BatchNorm2d, pretrained=True):
    model = DRNA(Bottleneck, [3, 4, 6, 3], BatchNorm=BatchNorm)
    if pretrained:
        model.load_state_dict(modelZoo.load_url(modelUrls['resnet50'], model_dir=modelDir))
    return model


def DRNC26(BatchNorm=nn.BatchNorm2d, pretrained=True):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='C', BatchNorm=BatchNorm)
    if pretrained:
        pretrained = modelZoo.load_url(modelUrls['drn-c-26'], model_dir=modelDir)
        del pretrained['fc.weight']
        del pretrained['fc.bias']
        model.load_state_dict(pretrained)
    return model


def DRNC42(BatchNorm=nn.BatchNorm2d, pretrained=True):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', BatchNorm=BatchNorm)
    if pretrained:
        pretrained = modelZoo.load_url(modelUrls['drn-c-42'], model_dir=modelDir)
        del pretrained['fc.weight']
        del pretrained['fc.bias']
        model.load_state_dict(pretrained)
    return model


def DRNC58(BatchNorm=nn.BatchNorm2d, pretrained=True):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', BatchNorm=BatchNorm)
    if pretrained:
        pretrained = modelZoo.load_url(modelUrls['drn-c-58'], model_dir=modelDir)
        del pretrained['fc.weight']
        del pretrained['fc.bias']
        model.load_state_dict(pretrained)
    return model


def DRNC22(BatchNorm=nn.BatchNorm2d, pretrained=True):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='D', BatchNorm=BatchNorm)
    if pretrained:
        pretrained = modelZoo.load_url(modelUrls['drn-d-22'], model_dir=modelDir)
        del pretrained['fc.weight']
        del pretrained['fc.bias']
        model.load_state_dict(pretrained)
    return model


def DRND24(BatchNorm=nn.BatchNorm2d, pretrained=True):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 2, 2], arch='D', BatchNorm=BatchNorm)
    if pretrained:
        pretrained = modelZoo.load_url(modelUrls['drn-d-24'], model_dir=modelDir)
        del pretrained['fc.weight']
        del pretrained['fc.bias']
        model.load_state_dict(pretrained)
    return model


def DRND38(BatchNorm=nn.BatchNorm2d, pretrained=True):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', BatchNorm=BatchNorm)
    if pretrained:
        pretrained = modelZoo.load_url(modelUrls['drn-d-38'], model_dir=modelDir)
        del pretrained['fc.weight']
        del pretrained['fc.bias']
        model.load_state_dict(pretrained)
    return model


def DRND40(BatchNorm=nn.BatchNorm2d, pretrained=True):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', BatchNorm=BatchNorm)
    if pretrained:
        pretrained = modelZoo.load_url(modelUrls['drn-d-40'], model_dir=modelDir)
        del pretrained['fc.weight']
        del pretrained['fc.bias']
        model.load_state_dict(pretrained)
    return model


def DRND54(BatchNorm=nn.BatchNorm2d, pretrained=True, outputStride=8):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', BatchNorm=BatchNorm, outputStride=outputStride)
    if pretrained:
        pretrained = modelZoo.load_url(modelUrls['drn-d-54'], model_dir=modelDir)
        del pretrained['fc.weight']
        del pretrained['fc.bias']
        model.load_state_dict(pretrained)
    return model


def DRND105(BatchNorm=nn.BatchNorm2d, pretrained=True):
    model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 1, 1], arch='D', BatchNorm=BatchNorm)
    if pretrained:
        pretrained = modelZoo.load_url(modelUrls['drn-d-105'], model_dir=modelDir)
        del pretrained['fc.weight']
        del pretrained['fc.bias']
        model.load_state_dict(pretrained)
    return model


if __name__ == "__main__":
    import torch

    m = DRND54(BatchNorm=nn.BatchNorm2d, pretrained=False, outputStride=8)
    inp = torch.rand(1, 3, 512, 512)
    outp, low = m(inp)
    print(outp.size())
    print(low.size())
