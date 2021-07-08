import torch.nn as nn
import torch.nn.functional as F

from modules.backbones.resnet import ResNet

class CenterNetResNetFPN(nn.Module):
  def __init__(self, backbone: ResNet):
    super(CenterNetResNetFPN, self).__init__()
    self.backbone = backbone

    self.conv1 = nn.Sequential(
      nn.Conv2d(256, 128, kernel_size=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True))
    self.conv2 = nn.Sequential(
      nn.Conv2d(128, 64, kernel_size=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True))
    self.conv3 = nn.Sequential(
      nn.Conv2d(64, 64, kernel_size=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True))

    self.lateral1 = nn.Conv2d(self.backbone.in_planes // 1, 256, kernel_size=1, padding='same', bias=False)
    self.lateral2 = nn.Conv2d(self.backbone.in_planes // 2, 256, kernel_size=1, padding='same', bias=False)
    self.lateral3 = nn.Conv2d(self.backbone.in_planes // 4, 128, kernel_size=1, padding='same', bias=False)
    self.lateral4 = nn.Conv2d(self.backbone.in_planes // 8,  64, kernel_size=1, padding='same', bias=False)

    self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, padding='valid', bias=False)
    self.smooth2 = nn.Conv2d(128, 128, kernel_size=3, padding='valid', bias=False)
    self.smooth3 = nn.Conv2d( 64,  64, kernel_size=3, padding='valid', bias=False)
    self.smooth4 = nn.Conv2d( 64,  64, kernel_size=3, padding='valid', bias=False)

  def _upsample_add(self, x, y):
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode='nearest') + y

  def forward(self, x):
    c2, c3, c4, c5 = self.backbone.extract_features(x)

    m5 = self.lateral1(c5)
    m4 = self.conv1(self._upsample_add(m5, self.lateral2(c4)))
    m3 = self.conv2(self._upsample_add(m4, self.lateral3(c3)))
    m2 = self.conv3(self._upsample_add(m3, self.lateral4(c2)))

    p5 = self.smooth1(m5)
    p4 = self.smooth2(m4)
    p3 = self.smooth3(m3)
    p2 = self.smooth4(m2)
    return p2, p3, p4, p5