import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding

class InvertedResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1, expansion_factor=6):
    super(InvertedResidualBlock, self).__init__()
    self.out_channels = out_channels
    self.in_channels = in_channels
    self.planes = in_channels * expansion_factor
    self.stride = stride

    self.conv1 = nn.Conv2d(
      in_channels=self.in_channels,
      out_channels=self.planes,
      kernel_size=1,
      stride=1,
      padding=0,
      bias=False)
    self.bn1 = nn.BatchNorm2d(self.planes)
    self.dw_conv2 = nn.Conv2d(
      in_channels=self.planes,
      out_channels=self.planes,
      kernel_size=3,
      stride=stride,
      padding=1,
      groups=self.planes,
      bias=False)
    self.bn2 = nn.BatchNorm2d(self.planes)
    self.conv3 = nn.Conv2d(
      in_channels=self.planes,
      out_channels=self.out_channels,
      kernel_size=1,
      stride=1,
      padding=0,
      bias=False)
    self.bn3 = nn.BatchNorm2d(self.out_channels)

  def forward(self, x):
    identity = x

    x = F.relu6(self.bn1(self.conv1(x)))
    x = F.relu6(self.bn2(self.dw_conv2(x)))
    x = self.bn3(self.conv3(x))

    if self.stride == 1 and self.in_channels == self.out_channels:
      x += identity

    return x

class MobileNetV2(nn.Module):
  def __init__(self, classes=1):
    super(MobileNetV2, self).__init__()
    self.classes = classes
    self.lateral_channels = [1280, 64, 32, 24]

    self.conv1 = nn.Conv2d(
      in_channels=3,
      out_channels=32,
      kernel_size=3,
      stride=2,
      padding=0,
      bias=False)
    self.bn1 = nn.BatchNorm2d(32)
    self.bottleneck1 = self._make_layer(in_channels=32,  out_channels=16,  stride=1, expansion_factor=1, num_blocks=1)
    self.bottleneck2 = self._make_layer(in_channels=16,  out_channels=24,  stride=2, expansion_factor=6, num_blocks=2)
    self.bottleneck3 = self._make_layer(in_channels=24,  out_channels=32,  stride=2, expansion_factor=6, num_blocks=3)
    self.bottleneck4 = self._make_layer(in_channels=32,  out_channels=64,  stride=2, expansion_factor=6, num_blocks=4)
    self.bottleneck5 = self._make_layer(in_channels=64,  out_channels=96,  stride=1, expansion_factor=6, num_blocks=3)
    self.bottleneck6 = self._make_layer(in_channels=96,  out_channels=160, stride=2, expansion_factor=6, num_blocks=3)
    self.bottleneck7 = self._make_layer(in_channels=160, out_channels=320, stride=1, expansion_factor=6, num_blocks=1)
    self.conv2 = nn.Conv2d(
      in_channels=320,
      out_channels=1280,
      kernel_size=1,
      stride=1,
      padding=0,
      bias=False)
    self.bn2 = nn.BatchNorm2d(1280)
    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.dropout = nn.Dropout(p=0.2, inplace=True)
    self.fc = nn.Linear(1280, self.classes)

  def _make_layer(self, in_channels, out_channels, stride, expansion_factor, num_blocks):
    layers = [
      InvertedResidualBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        expansion_factor=expansion_factor)]

    for _ in range(1, num_blocks):
      layers.append(
        InvertedResidualBlock(
          in_channels=out_channels,
          out_channels=out_channels,
          stride=1,
          expansion_factor=expansion_factor))
    
    return nn.Sequential(*layers)

  def extract_features(self, x):
    # First convolutional step
    x = F.relu(self.bn1(self.conv1(x)))

    # Execute linear bottlenecks
    x = self.bottleneck1(x)
    c2 = self.bottleneck2(x)
    c3 = self.bottleneck3(c2)
    c4 = self.bottleneck4(c3)
    x = self.bottleneck5(c4)
    x = self.bottleneck6(x)
    x = self.bottleneck7(x)

    # Last convolutional step
    c5 = F.relu(self.bn2(self.conv2(x)))
    return c2, c3, c4, c5

  def forward(self, x):
    # First convolutional step
    x = F.relu(self.bn1(self.conv1(x)))

    # Execute linear bottlenecks
    x = self.bottleneck1(x)
    x = self.bottleneck2(x)
    x = self.bottleneck3(x)
    x = self.bottleneck4(x)
    x = self.bottleneck5(x)
    x = self.bottleneck6(x)
    x = self.bottleneck7(x)

    # Last convolutional step
    x = F.relu(self.bn2(self.conv2(x)))

    x = self.avgpool(x)
    x = self.dropout(x)

    # Flatten and fully-connected
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x