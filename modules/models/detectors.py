import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

class FpnDetector(nn.Module):
  def __init__(self, fpn_output_channels=[32, 64, 128]):
    super().__init__()
    self.backbone = nn.ModuleList(self.backbone_layers())
    self.heads = nn.ModuleList(self.prediction_heads())

    self.fpn_output_channels = fpn_output_channels
    self.backbone_output_channels = self._get_layer_channels(*self.backbone)

    self.conv1 = nn.Sequential(
      nn.Conv2d(self.fpn_output_channels[2], self.fpn_output_channels[1], kernel_size=1, padding=1, bias=False),
      nn.BatchNorm2d(self.fpn_output_channels[1]),
      nn.ReLU(inplace=True))
    self.conv2 = nn.Sequential(
      nn.Conv2d(self.fpn_output_channels[1], self.fpn_output_channels[0], kernel_size=1, padding=1, bias=False),
      nn.BatchNorm2d(self.fpn_output_channels[0]),
      nn.ReLU(inplace=True))
    self.conv3 = nn.Sequential(
      nn.Conv2d(self.fpn_output_channels[0], self.fpn_output_channels[0], kernel_size=1, padding=1, bias=False),
      nn.BatchNorm2d(self.fpn_output_channels[0]),
      nn.ReLU(inplace=True))

    self.lateral1 = nn.Conv2d(self.backbone_output_channels[3], self.fpn_output_channels[2], kernel_size=1, padding='same', bias=False)
    self.lateral2 = nn.Conv2d(self.backbone_output_channels[2], self.fpn_output_channels[2], kernel_size=1, padding='same', bias=False)
    self.lateral3 = nn.Conv2d(self.backbone_output_channels[1], self.fpn_output_channels[1], kernel_size=1, padding='same', bias=False)
    self.lateral4 = nn.Conv2d(self.backbone_output_channels[0], self.fpn_output_channels[0], kernel_size=1, padding='same', bias=False)

    self.smooth1 = nn.Conv2d(self.fpn_output_channels[2], self.fpn_output_channels[2], kernel_size=3, padding='valid', bias=False)
    self.smooth2 = nn.Conv2d(self.fpn_output_channels[1], self.fpn_output_channels[1], kernel_size=3, padding='valid', bias=False)
    self.smooth3 = nn.Conv2d(self.fpn_output_channels[0], self.fpn_output_channels[0], kernel_size=3, padding='valid', bias=False)
    self.smooth4 = nn.Conv2d(self.fpn_output_channels[0], self.fpn_output_channels[0], kernel_size=3, padding='valid', bias=False)

  @abstractmethod
  def backbone_layers(self):
    pass

  @abstractmethod
  def prediction_heads(self):
    pass

  def _get_layer_channels(self, *layers):
    x = torch.zeros((1, 3, 224, 224))
    channels = []

    for layer in layers:
      x = layer(x)
      channels.append(x.shape[1])

    return channels

  def _upsample_add(self, x, y):
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode='nearest') + y

  def forward(self, x):
    c2 = self.backbone[0](x)
    c3 = self.backbone[1](c2)
    c4 = self.backbone[2](c3)
    c5 = self.backbone[3](c4)

    m5 = self.lateral1(c5)
    m4 = self.conv1(self._upsample_add(m5, self.lateral2(c4)))
    m3 = self.conv2(self._upsample_add(m4, self.lateral3(c3)))
    m2 = self.conv3(self._upsample_add(m3, self.lateral4(c2)))

    # p5 = self.smooth1(m5)
    # p4 = self.smooth2(m4)
    # p3 = self.smooth3(m3)
    p2 = self.smooth4(m2)

    predictions = []
    for head in self.heads:
      predictions.append(head(p2))

    return predictions