import os
import glob
from PIL import Image
from torchvision import transforms
import numpy as np
import time
from numpy.core.numeric import Inf
import torch
import torch.utils.data
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

kp_text_map = ["nose",
               "left_eye",
               "right_eye",
               "left_ear",
               "right_ear",
               "left_shoulder",
               "right_shoulder",
               "left_elbow",
               "right_elbow",
               "left_wrist",
               "right_wrist",
               "left_hip",
               "right_hip",
               "left_knee",
               "right_knee",
               "left_ankle",
               "right_ankle"]

def find_maximum(heatmap):
  rows = heatmap.shape[0]
  cols = heatmap.shape[1]

  max_row = 0
  max_col = 0
  max_val = -Inf

  for row in range(rows):
    for col in range(cols):
      val = heatmap[row, col]
      if val >= max_val:
        max_val = val
        max_row = row
        max_col = col

  return max_row, max_col, max_val

def read_images_as_tensors(paths):
  tensors = torch.zeros((len(paths), 3, 224, 224))

  preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  for i, path in enumerate(paths):
    input_image = Image.open(path)
    input_tensor = preprocess(input_image)
    tensors[i] = input_tensor

  return tensors

parser = argparse.ArgumentParser()
cfg = parser.parse_args()

if __name__ == '__main__':
  cfg.data_dir = 'datasets/coco2017'
  cfg.device = 'cuda'

  list_of_files = glob.glob('checkpoints/*.pt')
  cfg.model_file = max(list_of_files, key=os.path.getctime)
  model = torch.load(cfg.model_file)

  image_paths = glob.glob('eval/*')
  input_batch = read_images_as_tensors(image_paths).to(cfg.device)
  
  start = time.perf_counter()
  y = model(input_batch)
  end = time.perf_counter()
  print(end - start)

  num_inputs = len(image_paths)
  fig, ax = plt.subplots(num_inputs, 3)

  for i in range(num_inputs):
    hmaps = y[0][i].cpu().detach().numpy().astype(np.float32)
    keypoints = []

    for idx in range(hmaps.shape[0]):
      max_row, max_col, max_val = find_maximum(hmaps[idx])
      keypoint = (float(max_row) / hmaps.shape[1], float(max_col) / hmaps.shape[2])
      keypoints.append(keypoint)
    
    input_tensor = input_batch[i].cpu().detach()
    ax[i,0].imshow(input_tensor.permute(1, 2, 0))

    for idx, kp in enumerate(keypoints):
      cx = kp[1] * input_tensor.shape[2]
      cy = kp[0] * input_tensor.shape[1]
      ax[i,0].text(cx, cy - 3, kp_text_map[idx], horizontalalignment='center', verticalalignment='center')
      circle = Circle((cx, cy), 2)
      ax[i,0].add_patch(circle)

    ax[i,1].imshow(hmaps.sum(axis=0))
    ax[i,2].imshow(hmaps[0])

  plt.show()