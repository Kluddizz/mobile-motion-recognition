import numpy as np
import argparse
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt

from datasets import COCOKeypointDataset
from modules.models.pose_estimator import PoseEstimator

parser = argparse.ArgumentParser()
cfg = parser.parse_args()

def _reg_loss(regs, gt_regs, mask):
  mask = mask[:, :, None].expand_as(gt_regs).float()
  loss = sum(F.l1_loss)

if __name__ == '__main__':
  cfg.num_gpus = torch.cuda.device_count()
  cfg.data_dir = 'datasets/coco2017'
  cfg.epochs = 500
  cfg.batch_size = 48
  cfg.num_workers = 2
  cfg.device = 'cuda'
  cfg.lr = 5e-4

  train_dataset = COCOKeypointDataset(cfg.data_dir, split='train')
  train_loader = data.DataLoader(
    dataset=train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
    drop_last=True
  )

  sample = next(iter(train_dataset))
  img = np.array(sample['image'])

  plt.figure()
  plt.imshow(sample['hmap'][0])
  # plt.imshow(img.reshape(img.shape[1], img.shape[2], -1))
  plt.show()

  # model = PoseEstimator(17, 'mobilenetv2', fpn=True)
  # optimizer = torch.optim.Adam(model.parameters(), cfg.lr)

  # for epoch in range(cfg.epochs):
  #   model.to(cfg.device)
  #   model.train()

  #   for batch in train_loader:
  #     batch['image'].to(cfg.device)
  #     batch['keypoint'].to(cfg.device)
  #     outputs = model(batch['image'])
  #     kps = outputs[0]

