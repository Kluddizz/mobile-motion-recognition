import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU

class CenterNetHead(nn.Module):
  def __init__(self, in_channels, head_conv, classes):
    super(CenterNetHead, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True)
    self.conv2 = nn.Conv2d(head_conv, classes, kernel_size=1, stride=1, padding=0, bias=True)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    return x