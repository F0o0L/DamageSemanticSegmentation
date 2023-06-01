import torch
import torch.nn as nn
import torch.nn.functional as F
from MyModeling.Backbone import buildBackbone
from MyModeling.ASPP import buildASPP
from MyModeling.Decoder import buildDecoder
from typing import Optional


def weightsInit(net: nn.Module,
                initType: str = 'normal',
                initGain: float = 0.02) -> None:
    def initFunc(m):
        className = m.__class__.__name__
        if hasattr(m, 'weight') and className.find('Conv') != -1:
            if initType == 'normal':
                nn.init.normal_(m.weight.data, 0.0, initGain)
            elif initType == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=initGain)
            elif initType == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif initType == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=initGain)
            else:
                raise NotImplementedError('ini method [%s] is not implemented' % initType)
        elif className.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, initGain)
            nn.init.constant_(m.bias.data, 0.0)
        elif className.find('Linear') != -1:
            nn.init.normal_(m.weight, 0, initGain)
            nn.init.constant_(m.bias, 0)

    # Apply the ini function <init_func>
    print('Initialize network with %s type' % initType)
    net.apply(initFunc)


class DeeplabV3Plus(nn.Module):
    def __init__(self,
                 backbone: str,
                 ASPP: str,
                 preTrained: bool,
                 modify: bool,
                 dilations: list,
                 numClasses: int,
                 outputStride: Optional[int] = None,
                 lowLevelOutputStride: Optional[int] = None,
                 dilated: Optional[bool] = None) -> None:
        super(DeeplabV3Plus, self).__init__()

        # define backbone
        self.backbone = buildBackbone(backbone=backbone,
                                      preTrained=preTrained,
                                      outputStride=outputStride,
                                      lowLevelOutputStride=lowLevelOutputStride,
                                      dilated=dilated)
        if not preTrained:
            weightsInit(self.backbone, initType='kaiming')

        if backbone == 'Xception' or backbone == 'DRN':
            lowLevelOutputStride = 'None'
        setting = {
            # [backbone,aspp,lowLevelOutputStride]:[inChannel,lowLevelInChannel]
            'DRN' + 'None': [512, 256],
            'Xception' + 'None': [2048, 128],
            'MobileNetV3Small' + str(4): [576, 16],
            'MobileNetV3Small' + str(8): [576, 24],
            'MobileNetV3Small' + str(16): [576, 40],
            'MobileNetV3Large' + str(4): [960, 24],
            'MobileNetV3Large' + str(8): [960, 40],
            'MobileNetV3Large' + str(16): [960, 80],
            'MobileNetV2' + str(4): [1280, 24],
            'MobileNetV2' + str(8): [1280, 32],
            'MobileNetV2' + str(16): [1280, 64],
        }
        self.aspp = buildASPP(inChannel=setting[backbone + str(lowLevelOutputStride)][0],
                              mode=ASPP,
                              dilations=dilations,
                              modify=modify)
        self.decoder = buildDecoder(numClasses=numClasses,
                                    lowLevelInChannel=setting[backbone + str(lowLevelOutputStride)][1],
                                    mode=ASPP)

        weightsInit(self.aspp, initType='kaiming')
        weightsInit(self.decoder, initType='kaiming')

        # self.freezeBn()

    def forward(self, xxx):
        x, lowLevelFeat = self.backbone(xxx)
        x = self.aspp(x)
        x = self.decoder(lowLevelFeat, x)
        x = F.interpolate(x, size=xxx.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freezeBn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


if __name__ == "__main__":
    model = DeeplabV3Plus(backbone='DRN',
                          ASPP='ASPP',
                          preTrained=False,
                          modify=True,
                          dilations=[1, 3, 6, 9],
                          numClasses=3,
                          outputStride=16)
    model = model.cuda()
    model.eval()
    inp = torch.rand(1, 3, 192, 256)
    out = model(inp)
    print(out.size())
