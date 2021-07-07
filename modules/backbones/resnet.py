import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
  '''
  Implementation of the basic building block for ResNet-18/34 `[1]`.

  * `[1] Deep Residual Learning for Image Recognition, https://arxiv.org/pdf/1512.03385.pdf`
  '''
  expansion = 1

  def __init__(self, in_dim, in_planes, planes, stride=1):
    super(BasicBlock, self).__init__()

    self.conv1 = nn.Conv2d(in_dim, in_planes, kernel_size=3, stride=stride, bias=False, padding=1)
    self.bn1 = nn.BatchNorm2d(in_planes)
    self.conv2 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, bias=False, padding=1)
    self.bn2 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)

    self.downsample = nn.Sequential(
      nn.Conv2d(in_dim, in_planes * self.expansion, kernel_size=1, stride=stride),
      nn.BatchNorm2d(in_planes * self.expansion)
    )

  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out += self.downsample(x)
    out = self.relu(out)
    return out


class BottleneckBlock(nn.Module):
  '''
  Implementation of the bottleneck building block for ResNet-50/101/152 `[1]`.

  * `[1] Deep Residual Learning for Image Recognition, https://arxiv.org/pdf/1512.03385.pdf`
  '''
  expansion = 4

  def __init__(self, in_dim, in_planes, planes, stride=1):
    super(BottleneckBlock, self).__init__()

    self.conv1 = nn.Conv2d(in_dim, in_planes, kernel_size=1, stride=1, bias=False)
    self.bn1 = nn.BatchNorm2d(in_planes)
    self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, stride=stride, bias=False)
    self.bn2 = nn.BatchNorm2d(in_planes)
    self.conv3 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)

    self.downsample = nn.Sequential(
      nn.Conv2d(in_dim, in_planes * self.expansion, kernel_size=1, stride=stride),
      nn.BatchNorm2d(in_planes * self.expansion)
    )

  def forward(self, x):
    # 1x1, 64
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    # 3x3, 64
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    # 1x1, 256
    out = self.conv3(out)
    out = self.bn3(out)

    out += self.downsample(x)
    out = self.relu(out)

    return out

class ResNet(nn.Module):
  '''
  Implementation of ResNet `[1]`.

  * `[1] Deep Residual Learning for Image Recognition, https://arxiv.org/pdf/1512.03385.pdf`
  '''

  def __init__(self, block, layers=[]):
    super(ResNet, self).__init__()
    self.in_planes = 64

    self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(self.in_planes)

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.conv2 = self._make_layer(block,  64, layers[0], stride=1)
    self.conv3 = self._make_layer(block, 128, layers[1], stride=2)
    self.conv4 = self._make_layer(block, 256, layers[2], stride=2)
    self.conv5 = self._make_layer(block, 512, layers[3], stride=2)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

  def _make_layer(self, block, planes, number_blocks, stride=1):
    out_planes = planes * block.expansion

    layers = [
        block(
          self.in_planes,
          planes,
          out_planes,
          stride=stride
        )
    ]

    self.in_planes = out_planes

    for _ in range(1, number_blocks):
      layers.append(
        block(
          out_planes,
          planes,
          out_planes,
          stride=1
        )
      )

    return nn.Sequential(*layers)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.maxpool(out)

    out = self.conv2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.conv5(out)

    # out = self.avgpool(out)
    return out

def resnet18():
  return ResNet(
    BasicBlock,
    layers=[2, 2, 2, 2]
  )

def resnet34():
  return ResNet(
    BasicBlock,
    layers=[3, 4, 6, 3]
  )

def resnet50():
  return ResNet(
    BottleneckBlock,
    layers=[3, 4, 6, 3]
  )

def resnet101():
  return ResNet(
    BottleneckBlock,
    layers=[3, 4, 23, 3]
  )

def resnet152():
  return ResNet(
    BottleneckBlock,
    layers=[3, 8, 36, 3]
  )