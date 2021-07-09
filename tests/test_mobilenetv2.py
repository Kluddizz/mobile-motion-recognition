import torch
import unittest

from modules.backbones.mobilenetv2 import MobileNetV2

class TestMobileNetV2(unittest.TestCase):

  def test_output_shape(self):
    # Generate random image
    x = torch.rand((1, 3, 224, 224))

    # Run inference
    classes = 10
    network = MobileNetV2(classes=classes)
    output = network(x)

    self.assertEqual(output.shape, (1, classes))

  def test_extract_features(self):
    # Generate random image
    x = torch.rand((1, 3, 224, 224))

    # Run inference
    classes = 10
    network = MobileNetV2(classes=classes)
    c2, c3, c4, c5 = network.extract_features(x)

    self.assertEqual(c2.shape, (1, 24, 56, 56))
    self.assertEqual(c3.shape, (1, 32, 28, 28))
    self.assertEqual(c4.shape, (1, 64, 14, 14))
    self.assertEqual(c5.shape, (1, 1280, 7, 7))
