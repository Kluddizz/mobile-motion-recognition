import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import pycocotools.coco as coco

class COCOKeypointDataset(Dataset):
  def __init__(self, data_dir, split='train', img_size=(128, 128)):
    super(COCOKeypointDataset, self).__init__()
    self.data_dir = data_dir
    self.img_dir = os.path.join(data_dir, f'{split}2017')
    self.split = split
    self.img_size = img_size
    self.mean = np.array([0.40789654, 0.44719302, 0.47026115], np.float32)[None, None, :]
    self.std = np.array([0.28863828, 0.27408164, 0.27809835], np.float32)[None, None, :]

    annotation_file = os.path.join(self.data_dir, 'annotations', f'person_keypoints_{split}2017.json')
    self.coco = coco.COCO(annotation_file)
    self.image_ids = self.coco.getImgIds()
    self.num_samples = len(self.image_ids)
  
  def _process_image(self, img, mean, std):
    result = img.astype(np.float32) / 255.0
    result -= mean
    result /= std
    return result.transpose(2, 0, 1)

  def _gaussian2D(self, shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

  def _apply_gaussian_filter(self, img, center, radius=1):
    diameter = 2*radius+1
    gauss_filter = self._gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[1]), int(center[0])
    w, h = img.shape[1], img.shape[0]

    left, right = min(x, radius), min(w - x, radius + 1)
    top, bottom = min(y, radius), min(h - y, radius + 1)

    masked_img = img[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gauss_filter[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_img.shape) > 0:
      np.maximum(masked_img, masked_gaussian, out=masked_img)
    return img

  def __getitem__(self, index):
    img_id = self.image_ids[index]
    img_filename = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, img_filename)
    img = cv2.imread(img_path)
    img = self._process_image(img, self.mean, self.std)

    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    labels = np.array([ann['category_id'] for ann in anns])
    joints = np.array([ann['keypoints'] for ann in anns])

    joint_hmap = np.zeros((17, 128, 128), dtype=np.float32)

    for (joint, label) in zip(joints, labels):

      if label == 1: # Check if label is 'person'
        for i in range(0, joint_hmap.shape[0]):
          y = joint[i * 2 + 0].astype(np.int32)
          x = joint[i * 2 + 1].astype(np.int32)
          print(x, y)
          self._apply_gaussian_filter(joint_hmap[i], (y, x), radius=10)

    return {
      'image': img,
      'keypoints': joints,
      'hmap': joint_hmap
    }

  def __len__(self):
    return self.num_samples