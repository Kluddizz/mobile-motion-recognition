from modules.extractors.fpn import FPN
from modules.heads.centernet import CenterNetHead
from modules.backbones.resnet import ResNet, resnet101, resnet152, resnet18, resnet34, resnet50
from modules.backbones.mobilenetv2 import MobileNetV2
import torch.nn as nn

backbone_map = {
  'resnet18': resnet18,
  'resnet34': resnet34,
  'resnet50': resnet50,
  'resnet101': resnet101,
  'resnet152': resnet152,
  'mobilenetv2': MobileNetV2,
}

class PoseEstimator(nn.Module):
  def __init__(self, classes, backbone='resnet18', head_conv=64, fpn=False):
    super(PoseEstimator, self).__init__()
    backbone_factory = self._create_backbone_model(backbone)
    self.backbone = backbone_factory()
    self.filters = [self.backbone.lateral_channels[1], self.backbone.lateral_channels[2], self.backbone.lateral_channels[3]]

    if fpn:
      self.feature_extractor = FPN(backbone=self.backbone, lateral_channels=self.backbone.lateral_channels, filters=self.filters)
    else:
      self.feature_extractor = self.backbone
      self.deconv = self._make_deconv_layer(self.filters, [4, 4, 4])

    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.dropout = nn.Dropout(p=0.2, inplace=True)
    self.fc = nn.Linear(self.backbone.lateral_channels[3], classes*2)

  def _create_backbone_model(self, backbone):
    if not backbone_map.__contains__(backbone):
      raise Exception(f"The chosen backbone model is not supported. Select from {backbone_map.keys()}.")

    backbone_fn = backbone_map[backbone]
    return backbone_fn
    
  def _get_deconv_cfg(self, deconv_kernel):
    if deconv_kernel == 4:
      padding = 1
      output_padding = 0
    elif deconv_kernel == 3:
      padding = 1
      output_padding = 1
    elif deconv_kernel == 2:
      padding = 0
      output_padding = 0
    
    return deconv_kernel, padding, output_padding

  def _get_backbone_output_shape(self):
    if isinstance(self.feature_extractor, ResNet):
      output_shape = self.feature_extractor.in_planes
    elif isinstance(self.feature_extractor, MobileNetV2):
      output_shape = 1280

    return output_shape

  def _make_deconv_layer(self, filters, kernels):
    layers = []

    in_planes = self._get_backbone_output_shape()
    for i in range(len(filters)):
      kernel, padding, output_padding = self._get_deconv_cfg(kernels[i])

      layers.append(nn.ConvTranspose2d(
          in_channels=in_planes,
          out_channels=filters[i],
          kernel_size=kernel,
          stride=2,
          padding=padding,
          output_padding=output_padding,
          bias=False))
      layers.append(nn.BatchNorm2d(filters[i]))
      layers.append(nn.ReLU(inplace=True))
      in_planes = filters[i]
    
    return nn.Sequential(*layers)

  def forward(self, x):
    if isinstance(self.feature_extractor, FPN):
      x, _, _, _ = self.feature_extractor(x)
    else:
      _, _, _, x = self.feature_extractor.extract_features(x)
      x = self.deconv(x)

    x = self.avgpool(x)
    x = self.dropout(x)
    x = x.view(x.size(0), -1)
    return [self.fc(x)]

class CenterNetPoseEstimator(nn.Module):
  def __init__(self, classes, backbone='resnet18', head_conv=64, fpn=False):
    super(CenterNetPoseEstimator, self).__init__()
    backbone_factory = self._create_backbone_model(backbone)
    self.backbone = backbone_factory()
    self.filters = [self.backbone.lateral_channels[1], self.backbone.lateral_channels[2], self.backbone.lateral_channels[3]]

    if fpn:
      self.feature_extractor = FPN(backbone=self.backbone, lateral_channels=self.backbone.lateral_channels, filters=self.filters)
    else:
      self.feature_extractor = self.backbone
      self.deconv = self._make_deconv_layer(self.filters, [4, 4, 4])

    self.joint_heatmap = CenterNetHead(self.backbone.lateral_channels[-1], head_conv, classes)
    self.joint_locations = CenterNetHead(self.backbone.lateral_channels[-1], head_conv, classes * 2)
    self.joint_offset = CenterNetHead(self.backbone.lateral_channels[-1], head_conv, 2)

  def _create_backbone_model(self, backbone):
    if not backbone_map.__contains__(backbone):
      raise Exception(f"The chosen backbone model is not supported. Select from {backbone_map.keys()}.")

    backbone_fn = backbone_map[backbone]
    return backbone_fn
    
  def _get_deconv_cfg(self, deconv_kernel):
    if deconv_kernel == 4:
      padding = 1
      output_padding = 0
    elif deconv_kernel == 3:
      padding = 1
      output_padding = 1
    elif deconv_kernel == 2:
      padding = 0
      output_padding = 0
    
    return deconv_kernel, padding, output_padding

  def _get_backbone_output_shape(self):
    if isinstance(self.feature_extractor, ResNet):
      output_shape = self.feature_extractor.in_planes
    elif isinstance(self.feature_extractor, MobileNetV2):
      output_shape = 1280

    return output_shape

  def _make_deconv_layer(self, filters, kernels):
    layers = []

    in_planes = self._get_backbone_output_shape()
    for i in range(len(filters)):
      kernel, padding, output_padding = self._get_deconv_cfg(kernels[i])

      layers.append(nn.ConvTranspose2d(
          in_channels=in_planes,
          out_channels=filters[i],
          kernel_size=kernel,
          stride=2,
          padding=padding,
          output_padding=output_padding,
          bias=False))
      layers.append(nn.BatchNorm2d(filters[i]))
      layers.append(nn.ReLU(inplace=True))
      in_planes = filters[i]
    
    return nn.Sequential(*layers)

  def forward(self, x):
    if isinstance(self.feature_extractor, FPN):
      x, _, _, _ = self.feature_extractor(x)
    else:
      _, _, _, x = self.feature_extractor.extract_features(x)
      x = self.deconv(x)

    return [self.joint_heatmap(x), self.joint_locations(x), self.joint_offset(x)]