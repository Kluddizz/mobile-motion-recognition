import os
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import pycocotools.coco as coco

class COCOKeypointDataset(Dataset):
  def __init__(self, data_dir, split='train', img_size=(512, 512)):
    super(COCOKeypointDataset, self).__init__()
    self.data_dir = data_dir
    self.img_dir = os.path.join(data_dir, f'{split}2017')
    self.split = split
    # self.img_size = img_size
    self.hmap_size = (img_size[0] // 4, img_size[1] // 4)

    annotation_file = os.path.join(self.data_dir, 'annotations', f'person_keypoints_{split}2017.json')
    self.coco = coco.COCO(annotation_file)
    self.catIds = self.coco.getCatIds(catNms=['person'])
    self.image_ids = self.coco.getImgIds(catIds=self.catIds)
    self.num_samples = len(self.image_ids)

  def _read_process_image(self, img_path):
    image = Image.open(img_path)
    original_image_shape = (3, image.size[1], image.size[0])

    preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = image.convert('RGB')
    image = preprocess(image)
    return image, original_image_shape

  #
  # OpenCV way of reading and augmenting image data
  #
  # def _process_image(self, img_path, output_size, mean, std):
  #   img = cv2.imread(img_path)
  #   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  #   original_shape = img.shape
  #   img = img.astype(np.float32) / 255.0
  #   img = cv2.resize(img, (output_size[1], output_size[0]), interpolation=cv2.INTER_AREA)
  #   img -= mean
  #   img /= std
  #   return img.transpose(2, 0, 1), original_shape

  def _gaussian2D(self, shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

  def gaussian(self, img, pt, sigma=4):
    '''
    Source: https://github.com/princeton-vl/pose-hg-train/blob/master/src/pypose/draw.py
    '''
    # Draw a 2D gaussian

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0):
      # If not, just return the image as is
      return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

  def _generate_heatmaps(self, ann, input_size, output_size):
    hmaps = np.zeros((17, output_size[0], output_size[1]), dtype=np.float32)

    if ann:
      keypoints = ann['keypoints']

      for i in range(0, 17):
        x = int((keypoints[i * 3 + 0] / input_size[1]) * output_size[1])
        y = int((keypoints[i * 3 + 1] / input_size[0]) * output_size[0])
        confidence = keypoints[i * 3 + 2]

        if confidence > 0:
          hmap = self.gaussian(np.zeros(output_size), (x, y), sigma=2)
          hmaps[i, :, :] = cv2.resize(hmap, (output_size[1], output_size[0]), interpolation=cv2.INTER_AREA)

    return hmaps

  def _get_annotation_with_max_keypoints(self, anns):
    '''
    Returns the annotation with the maximum number of keypoints.
    '''
    max_kp = 0
    ann = None

    for a in anns:
      if a['num_keypoints'] >= max_kp:
        ann = a
        max_kp = a['num_keypoints']

    return ann

  def __getitem__(self, index):
    img_id = self.image_ids[index]
    img_filename = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, img_filename)
    # img, original_img_shape = self._process_image(img_path, (self.img_size[0], self.img_size[1]), self.mean, self.std)
    img, original_img_shape = self._read_process_image(img_path)

    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    ann = self._get_annotation_with_max_keypoints(anns)
    # hmap = self._generate_heatmaps(ann, original_img_shape[1:], (self.hmap_size[0], self.hmap_size[1]))
    hmap = self._generate_heatmaps(ann, original_img_shape[1:], (28, 28))

    return {
      'image': img,
      'hmap': hmap
    }

  def __len__(self):
    return self.num_samples