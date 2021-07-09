import torch
import unittest

from modules.models.pose_estimator import CenterNetPoseEstimator

class TestCenterNetPoseEstimator(unittest.TestCase):

  def test_filters_resnet18(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    net = CenterNetPoseEstimator(17, backbone='resnet18')

    self.assertEqual(net.filters, [256, 128, 64])

  def test_filters_mobilenetv2(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    net = CenterNetPoseEstimator(17, backbone='mobilenetv2')

    self.assertEqual(net.filters, [64, 32, 24])

  def test_filters_mobilenetv2_fpn(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    net = CenterNetPoseEstimator(17, backbone='mobilenetv2', fpn=True)

    self.assertEqual(net.filters, [64, 32, 24])

  def test_output_shape_resnet50(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    net = CenterNetPoseEstimator(17, backbone='resnet50')
    output = net(x)

    self.assertEqual(output[0].shape, (1, 17, 56, 56))
    self.assertEqual(output[1].shape, (1, 34, 56, 56))
    self.assertEqual(output[2].shape, (1, 2, 56, 56))

  def test_output_shape_resnet50_fpn(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    net = CenterNetPoseEstimator(17, backbone='resnet50', fpn=True)
    output = net(x)

    self.assertEqual(output[0].shape, (1, 17, 56, 56))
    self.assertEqual(output[1].shape, (1, 34, 56, 56))
    self.assertEqual(output[2].shape, (1, 2, 56, 56))

if __name__ == '__main__':
  unittest.main()