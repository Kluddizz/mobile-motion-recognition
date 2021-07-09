import torch
import unittest

from modules.models.pose_estimator import CenterNetPoseEstimator1

class TestCenterNetPoseEstimator(unittest.TestCase):

  def test_output_shapes(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    net = CenterNetPoseEstimator1(17, backbone='resnet50')
    output = net(x)

    self.assertEqual(output[0].shape, (1, 17, 56, 56))
    self.assertEqual(output[1].shape, (1, 34, 56, 56))
    self.assertEqual(output[2].shape, (1, 2, 56, 56))

if __name__ == '__main__':
  unittest.main()