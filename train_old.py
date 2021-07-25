from modules.losses import FocalLoss
import yaml
import os
import argparse
import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from munch import DefaultMunch

from datasets import COCOKeypointDataset
from modules.models.pose_estimator import MobileNetV2FpnCenterNet, MobileNetV3LargeFpnCenterNet, MobileNetV3SmallFpnCenterNet

model_map = {
  'mobilenet_v2_fpn_centernet': MobileNetV2FpnCenterNet,
  'mobilenet_v3_large_fpn_centernet': MobileNetV3LargeFpnCenterNet,
  'mobilenet_v3_small_fpn_centernet': MobileNetV3SmallFpnCenterNet,
}

def init_model(cfg):
  if cfg.continue_training:
    model = torch.load(f'{cfg.checkpoint_dir}/{cfg.name}.pt')
  else:
    model = model_map[cfg.model]()

  return model

def init_optimizer(cfg, model):
  lr = float(cfg.training.learning_rate)

  if cfg.training.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr)
  else:
    optimizer = torch.optim.Adam(model.parameters(), lr)

  return optimizer

def init_loss(cfg):
  if cfg.training.loss == 'focal':
    loss = FocalLoss()
  else:
    loss = torch.nn.MSELoss()

  return loss
  
def init_dataset(cfg):
  train_dataset = COCOKeypointDataset(cfg.dataset.data_dir, split='train', img_size=(224, 224))
  train_loader = data.DataLoader(
    dataset=train_dataset,
    batch_size=cfg.dataset.batch_size,
    shuffle=cfg.dataset.shuffle,
    num_workers=cfg.dataset.num_workers,
    pin_memory=True,
    drop_last=True
  )

  return train_loader

def train(cfg, model, train_loader, optimizer, loss):
  writer = SummaryWriter()

  for epoch in range(0, cfg.training.epochs):
    model.to(device)
    model.train()

    for batch_idx, batch in enumerate(train_loader):
      batch['image'] = batch['image'].to(device, non_blocking=True)
      batch['hmap'] = batch['hmap'].to(device, non_blocking=True)

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

      print(f'Epoch {epoch}/{cfg.training.epochs}, {batch_idx+1}/{len(train_loader)}: hmap_loss {hmap_loss}', end='\r')

    torch.save(model, os.path.join(cfg.training.checkpoint_dir, f'{cfg.name}.pt'))
    writer.add_scalar('Loss/train', hmap_loss, epoch)

  writer.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('experiment', type=str)
  parser.add_argument('--continue-training', action='store_true')
  args = parser.parse_args()

  # Read experiment configuration
  with open(f'experiments/{args.experiment}.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
    cfg = DefaultMunch.fromDict(cfg, None)
    cfg.continue_training = args.continue_training

  # Create necessary directories
  if not os.path.exists(cfg.training.checkpoint_dir):
    os.makedirs(cfg.training.checkpoint_dir)

  # Start training
  device        = 'cuda' if torch.cuda.is_available() else 'cpu'
  train_loader  = init_dataset(cfg)
  model         = init_model(cfg)
  optimizer     = init_optimizer(cfg, model)
  loss          = init_loss(cfg)
  train(cfg, model, train_loader, optimizer, loss)