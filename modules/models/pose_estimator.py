from modules.extractors.fpn import CenterNetResNetFPN
from modules.heads.centernet import CenterNetHead
from modules.backbones.resnet import ResNet, resnet101, resnet152, resnet18, resnet34, resnet50
import torch.nn as nn

backbone_map = {
  'resnet18': resnet18,
  'resnet34': resnet34,
  'resnet50': resnet50,
  'resnet101': resnet101,
  'resnet152': resnet152,
}

class CenterNetPoseEstimator2(nn.Module):
  def __init__(self, classes, backbone='resnet18', head_conv=64):
    super(CenterNetPoseEstimator2, self).__init__()
    backbone_factory = self._create_backbone_model(backbone)
    self.feature_extractor = CenterNetResNetFPN(backbone_factory())

    self.joint_heatmap = CenterNetHead(64, head_conv, classes)
    self.joint_locations = CenterNetHead(64, head_conv, classes * 2)
    self.joint_offset = CenterNetHead(64, head_conv, 2)

  def _create_backbone_model(self, backbone):
    if not backbone_map.__contains__(backbone):
      raise Exception(f"The chosen backbone model is not supported. Select from {backbone_map.keys()}.")

    backbone_fn = backbone_map[backbone]
    return backbone_fn

  def forward(self, x):
    p2, p3, p4, p5 = self.feature_extractor(x)
    return [self.joint_heatmap(p2), self.joint_locations(p2), self.joint_offset(p2)]

class CenterNetPoseEstimator1(nn.Module):
  def __init__(self, classes, backbone='resnet18', head_conv=64):
    super(CenterNetPoseEstimator1, self).__init__()
    backbone_factory = self._create_backbone_model(backbone)
    self.feature_extractor = backbone_factory()

    self.deconv = self._make_deconv_layer([256, 128, 64], [4, 4, 4])
    self.joint_heatmap = CenterNetHead(64, head_conv, classes)
    self.joint_locations = CenterNetHead(64, head_conv, classes * 2)
    self.joint_offset = CenterNetHead(64, head_conv, 2)

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
    x = self.feature_extractor(x)
    x = self.deconv(x)
    return [self.joint_heatmap(x), self.joint_locations(x), self.joint_offset(x)]