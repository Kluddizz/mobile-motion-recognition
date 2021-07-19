import time
import torch
from tabulate import tabulate
from modules.models.pose_estimator import CenterNetPoseEstimator
from modules.backbones.resnet import resnet101, resnet152, resnet18, resnet34, resnet50
from modules.backbones.mobilenetv2 import MobileNetV2

network_map = {
  'resnet18':                   resnet18(classes=17),
  'resnet34':                   resnet34(classes=17),
  'resnet50':                   resnet50(classes=17),
  'resnet101':                  resnet101(classes=17),
  'resnet152':                  resnet152(classes=17),
  'mobilenetv2':                MobileNetV2(classes=17),
  'centernet_resnet18':         CenterNetPoseEstimator(17, backbone='resnet18'),
  'centernet_resnet18_fpn':     CenterNetPoseEstimator(17, backbone='resnet18', fpn=True),
  'centernet_resnet34':         CenterNetPoseEstimator(17, backbone='resnet34'),
  'centernet_resnet34_fpn':     CenterNetPoseEstimator(17, backbone='resnet34', fpn=True),
  'centernet_resnet50':         CenterNetPoseEstimator(17, backbone='resnet50'),
  'centernet_mobilenetv2':      CenterNetPoseEstimator(17, backbone='mobilenetv2'),
  'centernet_mobilenetv2_fpn':  CenterNetPoseEstimator(17, backbone='mobilenetv2', fpn=True),
}

def measure_time(function, *args):
  start_time = time.perf_counter_ns()

  function(*args)

  end_time = time.perf_counter_ns()
  estimated_ms = (end_time - start_time) / 1000000.0
  return estimated_ms

if __name__ == '__main__':
  device = 'cuda'

  # Generate random image
  x = torch.rand((1, 3, 224, 224)).to(device)
  data = []

  for key in network_map:
    network = network_map[key]
    network.to(device)
    ms = measure_time(network, x)
    data.append([key, round(ms, 2), round(1000.0 / ms)])
    
  print(tabulate(data, headers=['model', 'ms', 'fps']))
    
    