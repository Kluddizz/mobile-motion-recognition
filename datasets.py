import torch.utils.data as data

class COCODataset(data.Dataset):
  def __init__(self, data_dir, img_size=512):
    super(COCODataset, self).__init__()
    self.data_dir = data_dir
    self.num_classes = 17
    self.stride = 4
    self.img_size = (img_size, img_size)
    self.fmap_size = (self.img_size[0] // self.stride, self.img_size[1] // self.stride)
    

