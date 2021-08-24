import os
import numpy as np
from PIL import Image

def read_motion2021_dataset(path):
    annotation_file = os.path.join(path, 'annotations.txt')
    labels_file = os.path.join(path, 'labels.txt')

    train_x = []
    train_y = []

    with open(labels_file, 'r') as f:
        num_classes = len(f.readlines())

    with open(annotation_file, 'r') as f:
        annotations = [a.rstrip('\n') for a in f.readlines()]

        for annotation in annotations:
            splitted = annotation.split(' ')
            image_url = os.path.join(path, splitted[0])
            label_idx = int(splitted[1])
            label = np.zeros(num_classes, dtype=np.float32)
            label[label_idx] = 1.0

            image = Image.open(image_url)
            image = np.array(image, dtype=np.float32) / 255.0

            train_x.append(image)
            train_y.append(label)

    return np.array(train_x), np.array(train_y)