import io
import os
import sys
import shutil
from typing import TypedDict
import requests
from zipfile import ZipFile

coco2017_urls = {
  'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
  'train2017':   'http://images.cocodataset.org/zips/train2017.zip',
  'val2017':     'http://images.cocodataset.org/zips/val2017.zip',
  'test2017':    'http://images.cocodataset.org/zips/test2017.zip',
}

def download(url, desc=''):
  result = bytes()
  response = requests.get(url, stream=True)
  total = response.headers.get('content-length')

  if total is None:
    result += response.content
  else:
    downloaded = 0
    total = int(total)

    for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
      done = int(50 * downloaded / total)
      perc = int(downloaded / total)

      downloaded += len(data)
      result += data

      sys.stdout.write('\r[{}{}] {}% {}'.format('█' * done, '.' * (50-done), perc, desc))
      sys.stdout.flush()

  sys.stdout.write('\n')

  return result

def download_dataset_zips(base_dir, urls: TypedDict):
  '''
  Memory efficient download and loading of ZIP files
  '''
  for key in urls.keys():
    data_dir = os.path.join(base_dir, key)

    if not os.path.exists(data_dir):
      r = download(urls[key], key)

      with ZipFile(io.BytesIO(r)) as zip_file:
        yield zip_file

def download_dataset(base_dir, urls: TypedDict):
  zip_files = download_dataset_zips(base_dir, urls)

  for zip_file in zip_files:
    zip_file.extractall(base_dir)

if __name__ == '__main__':
  base_dir = './datasets/coco2017'
  
  # Create the dataset directory.
  if not os.path.exists(base_dir):
    os.makedirs(base_dir)

  # Download the dataset.
  download_dataset(base_dir, coco2017_urls)