import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from modules.models.movenet import MotionNet

def read_dataset_generator(num_classes, pose_estimator):
  with open('datasets/motions2021/annotations.txt', 'r') as f:
    annotations = [a.rstrip("\n") for a in f.readlines()]

  train_x = []
  train_y = []

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

    # Create empty buffer to store video frames
    keypoints = np.empty((num_frames, 17, 3), dtype=np.float32)

    # Fill video buffer with frame data
    for i in range(num_frames):
      ret, frame = cap.read()

      if not ret:
        break

      # Convert frame into correct shape and format and store it
      frame = cv2.resize(frame, (192, 192), interpolation=cv2.INTER_AREA)
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame = tf.convert_to_tensor(frame, dtype=tf.int32)
      frame = tf.expand_dims(frame, 0)

      keypoints[i] = pose_estimator(frame)['output_0'][0][0]

    yield keypoints, label

def read_dataset(num_classes, pose_estimator):
  with open('datasets/motions2021/annotations.txt', 'r') as f:
    annotations = [a.rstrip("\n") for a in f.readlines()]

  train_x = []
  train_y = []

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

    # Create empty buffer to store video frames
    keypoints = np.empty((num_frames, 17, 3), dtype=np.float32)

    # Fill video buffer with frame data
    for i in range(num_frames):
      ret, frame = cap.read()

      if not ret:
        break

      # Convert frame into correct shape and format and store it
      frame = cv2.resize(frame, (192, 192), interpolation=cv2.INTER_AREA)
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame = tf.convert_to_tensor(frame, dtype=tf.int32)
      frame = tf.expand_dims(frame, 0)

      keypoints[i] = pose_estimator(frame)['output_0'][0][0]

    # yield keypoints, label
    train_x.append(keypoints)
    train_y.append(label)

  return train_x, train_y

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', type=int, default=32)
  parser.add_argument('--epochs', type=int, default=50)
  parser.add_argument('--use-dataset-generator', action='store_true')
  args = parser.parse_args()

  with open('datasets/motions2021/labels.txt', 'r') as f:
    labels = [l.rstrip("\n") for l in f.readlines()]

  pose_model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
  pose_estimator = pose_model.signatures['serving_default']

  num_classes = len(labels)
  model = MotionNet(num_classes)

  if args.use_dataset_generator:
    train_dataset = tf.data.Dataset.from_generator(
      lambda: read_dataset_generator(num_classes=len(labels), pose_estimator=pose_estimator),
      output_types=(tf.int32, tf.int32)
    ).batch(args.batch_size).shuffle(4)
  else:
    train_x, train_y = read_dataset(num_classes=len(labels), pose_estimator=pose_estimator)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(args.batch_size).shuffle(4)

  num_batches = int(tf.data.experimental.cardinality(train_dataset).numpy())
  classifier_loss = tf.keras.losses.CategoricalCrossentropy()
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

  checkpoint_path = 'checkpoints/motionnet'

  model.compile(optimizer=optimizer, loss=classifier_loss, metrics=['accuracy'])

  if args.use_dataset_generator:
    for epoch in range(args.epochs):
      for idx, batch in enumerate(train_dataset):
        loss = model.train_on_batch(batch[0], batch[1])

        if epoch % (100*num_batches):
          tf.keras.models.save_model(model, checkpoint_path, include_optimizer=False)

        print(f'epoch {epoch + 1}/{args.epochs}, batch {idx + 1}, loss: {loss}')
  else:
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_path,
      verbose=True,
      save_freq=100*num_batches
    )

    model.fit(
      train_dataset,
      epochs=args.epochs,
      shuffle=True,
      callbacks=[cp_callback]
    )
