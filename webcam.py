import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

with open('datasets/motions2021/labels.txt', 'r') as f:
  labels = [l.rstrip("\n") for l in f.readlines()]

keypoints_buffer = np.zeros((1, 60, 17, 3), dtype=np.float32)
movenet_model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = movenet_model.signatures['serving_default']
motionnet = tf.keras.models.load_model('checkpoints/motionnet')
capture = cv2.VideoCapture(0)

while True:
  ret, frame = capture.read()
  img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
  x = tf.convert_to_tensor(img, dtype=tf.float32)
  x = tf.expand_dims(x, 0)
  x = tf.cast(tf.image.resize_with_pad(x, 192, 192), dtype=tf.int32)

  outputs = movenet(x)
  keypoints = outputs['output_0'][0][0]

  keypoints_buffer = np.roll(keypoints_buffer, -1, axis=1)
  keypoints_buffer[0, -1, :, :] = keypoints

  predicted_action = motionnet(keypoints_buffer)
  predicted_action_idx = np.argmax(predicted_action)
  label_text = labels[predicted_action_idx]

  for idx, kp in enumerate(keypoints):
    cx = int(kp[1] * img.shape[1])
    cy = int(kp[0] * img.shape[0])
    frame = cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
    frame = cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

  cv2.imshow('Video', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

capture.release()
cv2.destroyAllWindows()