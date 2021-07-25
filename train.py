import os
import cv2
import argparse
import numpy as np
import tensorflow as tf

from modules.models.movenet import MotionNet

def read_dataset(num_classes):
  with open('datasets/motions2021/annotations.txt', 'r') as f:
    annotations = [a.rstrip("\n") for a in f.readlines()]

  for annotation in annotations:
    # Annotation format is: <video_url> <label>
    splitted = annotation.split(' ')
    video_url = os.path.join('datasets', splitted[0])
    label_idx = int(splitted[1])
    label = np.zeros(num_classes, dtype=np.float32)
    label[label_idx] = 1.0

    # Load video file into memory
    cap = cv2.VideoCapture(video_url)

    # Read properties of video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_channels = 3

    # Create empty buffer to store video frames
    video = np.empty((num_frames, 192, 192, num_channels), dtype=np.int32)

    # Fill video buffer with frame data
    for i in range(num_frames):
      ret, frame = cap.read()

      if not ret:
        break

      # Convert frame into correct shape and format
      frame = cv2.resize(frame, (192, 192), interpolation=cv2.INTER_AREA)
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      # Normalize frame channels and store it into the frame buffer
      video[i] = frame

    yield video, label

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', type=int, default=32)
  parser.add_argument('--epochs', type=int, default=50)
  args = parser.parse_args()

  with open('datasets/motions2021/labels.txt', 'r') as f:
    labels = [l.rstrip("\n") for l in f.readlines()]

  train_dataset = tf.data.Dataset.from_generator(
    lambda: read_dataset(num_classes=len(labels)),
    output_types=(tf.int32, tf.int32)
  ).batch(args.batch_size)

  num_classes = len(labels)
  model = MotionNet(num_classes)

  classifier_loss = tf.keras.losses.CategoricalCrossentropy()
  optimizer = tf.keras.optimizers.Adam()

  for epoch in range(args.epochs):
    for idx, batch in enumerate(train_dataset):
      with tf.GradientTape() as tape:
        y = model(batch[0])
        loss = classifier_loss(batch[1], y)
      
      gradient = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradient, model.trainable_variables))
      print(f'epoch {epoch + 1}/{args.epochs}, batch {idx + 1}/{train_dataset._batch_size}, loss: {loss}')
