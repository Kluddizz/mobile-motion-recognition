import time
import torch
from modules.backbones.resnet import resnet101, resnet152, resnet18, resnet34, resnet50
from modules.models.pose_estimator import CenterNetPoseEstimator1, CenterNetPoseEstimator2

network_map = {
  'centernet_resnet50':     CenterNetPoseEstimator1(17, backbone='resnet50'),
  'centernet_resnet50_fpn': CenterNetPoseEstimator2(17, backbone='resnet50')
}

def measure_time(function, *args):
  start_time = time.time()

  function(*args)

  end_time = time.time()
  estimated_ms = (end_time - start_time) * 1000.0
  return estimated_ms

if __name__ == '__main__':
  # Generate random image
  x = torch.rand((1, 3, 224, 224))

  for key in network_map:
    network = network_map[key]
    ms = measure_time(network, x)
    print(f'{key}:\t{round(ms, 2)} ms')
