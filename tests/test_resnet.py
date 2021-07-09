import torch
import unittest
from modules.backbones.resnet import resnet101, resnet152, resnet18, resnet34, resnet50


class TestResNet(unittest.TestCase):

  def test_lateral_channels_resnet18(self):
    net = resnet18()
    self.assertEqual(net.lateral_channels, [512, 256, 128, 64])

  def test_output_shape_resnet18(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    # Run inference
    classes = 10
    net = resnet18(classes=classes)
    out = net(x)

    self.assertEqual(out.shape, (1, classes))

  def test_extract_features_resnet18(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    # Run inference
    net = resnet18()
    c2, c3, c4, c5 = net.extract_features(x)

    self.assertEqual(c2.shape, (1, 64, 56, 56))
    self.assertEqual(c3.shape, (1, 128, 28, 28))
    self.assertEqual(c4.shape, (1, 256, 14, 14))
    self.assertEqual(c5.shape, (1, 512, 7, 7))

  def test_output_shape_resnet34(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    # Run inference
    classes = 10
    net = resnet34(classes=classes)
    out = net(x)

    self.assertEqual(out.shape, (1, classes))

  def test_extract_features_resnet34(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    # Run inference
    net = resnet34()
    c2, c3, c4, c5 = net.extract_features(x)

    self.assertEqual(c2.shape, (1, 64, 56, 56))
    self.assertEqual(c3.shape, (1, 128, 28, 28))
    self.assertEqual(c4.shape, (1, 256, 14, 14))
    self.assertEqual(c5.shape, (1, 512, 7, 7))

  def test_output_shape_resnet50(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    # Run inference
    classes = 10
    net = resnet50(classes=classes)
    out = net(x)

    self.assertEqual(out.shape, (1, classes))

  def test_extract_features_resnet50(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    # Run inference
    net = resnet50()
    c2, c3, c4, c5 = net.extract_features(x)

    self.assertEqual(c2.shape, (1,  256, 56, 56))
    self.assertEqual(c3.shape, (1,  512, 28, 28))
    self.assertEqual(c4.shape, (1, 1024, 14, 14))
    self.assertEqual(c5.shape, (1, 2048, 7, 7))

  def test_output_shape_resnet101(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    # Run inference
    classes = 10
    net = resnet101(classes=classes)
    out = net(x)

    self.assertEqual(out.shape, (1, classes))

  def test_extract_features_resnet101(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    # Run inference
    net = resnet101()
    c2, c3, c4, c5 = net.extract_features(x)

    self.assertEqual(c2.shape, (1,  256, 56, 56))
    self.assertEqual(c3.shape, (1,  512, 28, 28))
    self.assertEqual(c4.shape, (1, 1024, 14, 14))
    self.assertEqual(c5.shape, (1, 2048, 7, 7))

  def test_output_shape_resnet152(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    # Run inference
    classes = 10
    net = resnet152(classes=classes)
    out = net(x)

    self.assertEqual(out.shape, (1, classes))


if __name__ == '__main__':
  unittest.main()