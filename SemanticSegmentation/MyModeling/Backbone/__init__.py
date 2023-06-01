import torch.nn as nn
from MyModeling.Backbone import DRN
from MyModeling.Backbone import MyXception
from MyModeling.Backbone import MyMobileNetV3
from MyModeling.Backbone import MyMobileNetV2
from typing import Optional


def buildBackbone(backbone: str,
                  preTrained: bool,
                  outputStride: Optional[int] = None,
                  lowLevelOutputStride: Optional[int] = None,
                  dilated: Optional[bool] = None) -> nn.Module:
    if backbone == 'Xception':
        return MyXception.AlignedXception(preTrained=preTrained,outputStride=outputStride)
    elif backbone == 'DRN':
        return DRN.DRND54(pretrained=preTrained,outputStride=outputStride)
    elif backbone == 'MobileNetV3Small':
        return MyMobileNetV3.mobileNetV3Small(outputStride=outputStride,
                                              lowLevelOutputStride=lowLevelOutputStride,
                                              preTrained=preTrained,
                                              dilated=dilated)
    elif backbone == 'MobileNetV3Large':
        return MyMobileNetV3.mobileNetV3Large(outputStride=outputStride,
                                              lowLevelOutputStride=lowLevelOutputStride,
                                              preTrained=preTrained,
                                              dilated=dilated)
    elif backbone == 'MobileNetV2':
        return MyMobileNetV2.mobileNetV2(outputStride=outputStride,
                                         lowLevelOutputStride=lowLevelOutputStride,
                                         preTrained=preTrained,
                                         dilated=dilated)
    else:
        raise ValueError(
            "backbone should be one of 'DRN','Xception','MobileNetV3Small','MobileNetV3Large' or 'MobileNetV2'")
