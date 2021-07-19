import torch
import unittest

from modules.models.pose_estimator import PoseEstimator

class TestCenterNetPoseEstimator(unittest.TestCase):

  def test_output_shape_pose_estimator(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    # Define network and run inference.
    net = PoseEstimator()
    output = net(x)

    # The network outputs uses one prediction head to predict heatmaps for all
    # 17 keypoints of an human.
    self.assertEqual(output[0].shape, (1, 17, 56, 56))

if __name__ == '__main__':
  unittest.main()