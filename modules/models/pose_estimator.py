from modules.heads.centernet import CenterNetHead
from torchvision.models.mobilenetv2 import mobilenet_v2
import torch.nn as nn
import torch.nn.functional as F

class PoseEstimator(nn.Module):
  def __init__(self):
    super(PoseEstimator, self).__init__()

    self.backbone_model = mobilenet_v2(pretrained=True)
    features = self.backbone_model.features

    for param in features.parameters():
      param.requires_grad = False

    # movenet bottleneck layers
    self.mn_layer1 = nn.Sequential(*features[0:4])
    self.mn_layer2 = nn.Sequential(*features[4:7])
    self.mn_layer3 = nn.Sequential(*features[7:11])
    self.mn_layer4 = nn.Sequential(*features[11:19])

    self.conv1 = nn.Sequential(
      nn.Conv2d(64, 32, kernel_size=1, padding=1, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True))
    self.conv2 = nn.Sequential(
      nn.Conv2d(32, 24, kernel_size=1, padding=1, bias=False),
      nn.BatchNorm2d(24),
      nn.ReLU(inplace=True))
    self.conv3 = nn.Sequential(
      nn.Conv2d(24, 24, kernel_size=1, padding=1, bias=False),
      nn.BatchNorm2d(24),
      nn.ReLU(inplace=True))

    self.lateral1 = nn.Conv2d(1280, 64, kernel_size=1, padding='same', bias=False)
    self.lateral2 = nn.Conv2d(  64, 64, kernel_size=1, padding='same', bias=False)
    self.lateral3 = nn.Conv2d(  32, 32, kernel_size=1, padding='same', bias=False)
    self.lateral4 = nn.Conv2d(  24, 24, kernel_size=1, padding='same', bias=False)

    self.smooth1 = nn.Conv2d(64, 64, kernel_size=3, padding='valid', bias=False)
    self.smooth2 = nn.Conv2d(32, 32, kernel_size=3, padding='valid', bias=False)
    self.smooth3 = nn.Conv2d(24, 24, kernel_size=3, padding='valid', bias=False)
    self.smooth4 = nn.Conv2d(24, 24, kernel_size=3, padding='valid', bias=False)
    
    self.joint_heatmap = CenterNetHead(24, 64, 17)

  def _upsample_add(self, x, y):
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode='nearest') + y

  def forward(self, x):
    c2 = self.mn_layer1(x)
    c3 = self.mn_layer2(c2)
    c4 = self.mn_layer3(c3)
    c5 = self.mn_layer4(c4)

    m5 = self.lateral1(c5)
    m4 = self.conv1(self._upsample_add(m5, self.lateral2(c4)))
    m3 = self.conv2(self._upsample_add(m4, self.lateral3(c3)))
    m2 = self.conv3(self._upsample_add(m3, self.lateral4(c2)))

    #p5 = self.smooth1(m5)
    #p4 = self.smooth2(m4)
    #p3 = self.smooth3(m3)
    p2 = self.smooth4(m2)
    return [self.joint_heatmap(p2)]