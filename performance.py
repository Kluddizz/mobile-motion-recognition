import time
import torch
import functools
from tabulate import tabulate
from modules.models.pose_estimator import PoseEstimator

network_map = {
  'mobilenetv2_fpn_centernet':  PoseEstimator(),
}

@functools.lru_cache(maxsize=None)
def measure_time(function, *args):
  start_time = time.perf_counter_ns()

  function(*args)

  end_time = time.perf_counter_ns()
  estimated_ms = (end_time - start_time) / 1000000.0
  return estimated_ms

if __name__ == '__main__':
  device = 'cpu'

  # Generate random image
  x = torch.rand((1, 3, 224, 224)).to(device)
  data = []

  for key in network_map:
    network = network_map[key]
    network.to(device)

    ms = 0.0

    for i in range(10):
      measure_time.cache_clear()
      ms += measure_time(network, x)

    ms /= 10.0

    data.append([key, round(ms, 2), round(1000.0 / ms)])
    
  print(tabulate(data, headers=['model', 'ms', 'fps']))
    
    