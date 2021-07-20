import time
import torch
import functools
import argparse
from numpy.core.numeric import Inf
from tabulate import tabulate
from modules.models.pose_estimator import MobileNetV2FpnCenterNet, MobileNetV3LargeFpnCenterNet, MobileNetV3SmallFpnCenterNet

parser = argparse.ArgumentParser()
parser.add_argument('device')
cfg = parser.parse_args()

network_map = {
  'mobilenet_v2_fpn_centernet':  MobileNetV2FpnCenterNet(),
  'mobilenet_v3_large_fpn_centernet':  MobileNetV3LargeFpnCenterNet(),
  'mobilenet_v3_small_fpn_centernet':  MobileNetV3SmallFpnCenterNet(),
}

@functools.lru_cache(maxsize=None)
def measure_time(function, *args):
  start_time = time.perf_counter_ns()

  function(*args)

  end_time = time.perf_counter_ns()
  estimated_ms = (end_time - start_time) / 1000000.0
  return estimated_ms

if __name__ == '__main__':
  # Generate random image
  x = torch.rand((1, 3, 224, 224)).to(cfg.device)
  data = []

  for key in network_map:
    network = network_map[key]
    network = network.to(cfg.device)

    best_ms = Inf

    for i in range(10):
      measure_time.cache_clear()
      ms = measure_time(network, x)

      if ms < best_ms:
        best_ms = ms

    data.append([key, round(best_ms, 2), round(1000.0 / best_ms)])
    
  print(tabulate(data, headers=['model', 'ms', 'fps']))
    
    