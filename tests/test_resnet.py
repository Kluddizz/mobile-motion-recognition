import torch
import unittest
from modules.backbones.resnet import resnet101, resnet152, resnet18, resnet34, resnet50


class TestResNet(unittest.TestCase):

  def test_output_shape_resnet18(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    # Run inference
    net = resnet18()
    out = net(x)

    assert out.shape == (1, 512, 7, 7)

  def test_output_shape_resnet34(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    # Run inference
    net = resnet34()
    out = net(x)

    assert out.shape == (1, 512, 7, 7)

  def test_output_shape_resnet50(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    # Run inference
    net = resnet50()
    out = net(x)

    assert out.shape == (1, 2048, 7, 7)

  def test_output_shape_resnet101(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    # Run inference
    net = resnet101()
    out = net(x)

    assert out.shape == (1, 2048, 7, 7)

  def test_output_shape_resnet152(self):
    # Create random image
    x = torch.rand(1, 3, 224, 224)

    # Run inference
    net = resnet152()
    out = net(x)

    assert out.shape == (1, 2048, 7, 7)


if __name__ == '__main__':
  unittest.main()