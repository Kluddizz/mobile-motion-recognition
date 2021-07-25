import torch.nn as nn
from torchvision.models.mobilenetv2 import mobilenet_v2
from torchvision.models.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small
from modules.models.detectors import FpnDetector
from modules.heads.centernet import CenterNetHead

class MobileNetV3LargeFpnCenterNet(FpnDetector):
  def __init__(self):
    super().__init__(fpn_output_channels=[40, 80, 160])

  def backbone_layers(self):
    backbone_model = mobilenet_v3_large(pretrained=True)
    features = backbone_model.features

    for param in features.parameters():
      param.requires_grad = False

    layer1 = nn.Sequential(*features[0:5])
    layer2 = nn.Sequential(*features[5:8])
    layer3 = nn.Sequential(*features[8:14])
    layer4 = nn.Sequential(*features[14:17])
    return layer1, layer2, layer3, layer4

  def prediction_heads(self):
    return [CenterNetHead(40, 64, 17)]

class MobileNetV3SmallFpnCenterNet(FpnDetector):
  def __init__(self):
    super().__init__(fpn_output_channels=[24, 32, 64])

  def backbone_layers(self):
    backbone_model = mobilenet_v3_small(pretrained=True)
    features = backbone_model.features

    for param in features.parameters():
      param.requires_grad = False

    layer1 = nn.Sequential(*features[0:3])
    layer2 = nn.Sequential(*features[3:5])
    layer3 = nn.Sequential(*features[5:10])
    layer4 = nn.Sequential(*features[10:13])
    return layer1, layer2, layer3, layer4

  def prediction_heads(self):
    return [CenterNetHead(24, 64, 17)]

class MobileNetV2FpnCenterNet(FpnDetector):
  def __init__(self):
    super().__init__(fpn_output_channels=[24, 32, 64])

  def backbone_layers(self):
    backbone_model = mobilenet_v2(pretrained=True)
    features = backbone_model.features

    for param in features.parameters():
      param.requires_grad = False

    layer1 = nn.Sequential(*features[0:4])
    layer2 = nn.Sequential(*features[4:7])
    layer3 = nn.Sequential(*features[7:11])
    layer4 = nn.Sequential(*features[11:19])
    return layer1, layer2, layer3, layer4

  def prediction_heads(self):
    return [CenterNetHead(24, 64, 17)]