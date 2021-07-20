import numpy as np
import os
import glob
import time
import datetime
import argparse
import torch
import torch.utils.data as data

from datasets import COCOKeypointDataset
from modules.models.pose_estimator import MobileNetV2FpnCenterNet

parser = argparse.ArgumentParser()
parser.add_argument('--continue-training', type=str, default=False)
parser.add_argument('--model-file', type=str, default=None)
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
parser.add_argument('--data-dir', type=str, default='datasets/coco2017')
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)
cfg = parser.parse_args()

def _neg_loss(preds, targets):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
      Arguments:
      preds (B x c x h x w)
      gt_regr (B x c x h x w)
  '''
  pos_inds = targets.eq(1).float()
  neg_inds = targets.lt(1).float()

  neg_weights = torch.pow(1 - targets, 4)

  loss = 0
  for pred in preds:
    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
      loss = loss - neg_loss
    else:
      loss = loss - (pos_loss + neg_loss) / num_pos
  return loss / len(preds)

if __name__ == '__main__':
  if not torch.cuda.is_available():
    print('Error loading cuda')
    exit()

  if not os.path.exists(cfg.checkpoint_dir):
    os.makedirs(cfg.checkpoint_dir)

  train_dataset = COCOKeypointDataset(cfg.data_dir, split='train', img_size=(224, 224))
  train_loader = data.DataLoader(
    dataset=train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
    drop_last=True
  )

  if cfg.continue_training:
    list_of_files = glob.glob('checkpoints/*.pt')
    cfg.model_file = max(list_of_files, key=os.path.getctime)
    model_name = os.path.splitext(os.path.basename(cfg.model_file))[0]
    cfg.start_epoch = int(model_name.split('_')[1]) + 1

    print(f'load model from "{cfg.model_file}" and continue with epoch {cfg.start_epoch}')
    model = torch.load(cfg.model_file)
  elif cfg.model_file is None:
    # model = CenterNetPoseEstimator(17, 'mobilenetv2', fpn=True)
    # model = MobileNetV2GAN(17, pretrained=True)
    model = MobileNetV2FpnCenterNet()
  else:
    model = torch.load(cfg.model_file)
    print(f'load model from "{cfg.model_file}" and continue with epoch {cfg.start_epoch}')

  optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
  loss = torch.nn.MSELoss()
  # loss = _neg_loss

  estimated_batch_times = np.zeros((200), dtype=np.float32)

  for epoch in range(cfg.start_epoch, cfg.epochs):
    model.to(cfg.device)
    model.train()

    total_batches = len(train_loader)
    start_time_batch = time.perf_counter()

    for batch_idx, batch in enumerate(train_loader):
      batch['image'] = batch['image'].to(cfg.device, non_blocking=True)
      batch['hmap'] = batch['hmap'].to(cfg.device, non_blocking=True)

      # Inference
      x = batch['image']
      outputs = model(x)

      # Calculating the loss
      hmap = outputs[0]
      hmap_loss = loss(hmap, batch['hmap'])

      # Backpropergate the loss
      optimizer.zero_grad()
      hmap_loss.backward()
      optimizer.step()

      end_time_batch = time.perf_counter()
      estimated_batch_times[0] = end_time_batch - start_time_batch
      start_time_batch = end_time_batch

      number_left_batches = total_batches - (batch_idx + 1)
      estimated_batch_times = np.roll(estimated_batch_times, 1)

      time_batch_avg = estimated_batch_times.sum() / len(estimated_batch_times)
      time_left = time_batch_avg * number_left_batches

      print(f'Epoch {epoch}/{cfg.epochs}, {batch_idx+1}/{len(train_loader)}: hmap_loss {hmap_loss}\t[{datetime.timedelta(seconds=time_left)}]', end='\r')

    torch.save(model, os.path.join(cfg.checkpoint_dir, f'model_{epoch}.pt'))