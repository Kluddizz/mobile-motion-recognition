import cv2
import tensorflow as tf
import tensorflow_hub as hub
from modules.models.movenet import MotionNet

movenet = MotionNet()
capture = cv2.VideoCapture(0)

while True:
  ret, frame = capture.read()
  img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
  x = tf.convert_to_tensor(img, dtype=tf.float32)
  x = tf.expand_dims(x, 0)
  x = tf.cast(tf.image.resize_with_pad(x, 192, 192), dtype=tf.int32)

  outputs = movenet(x)
  keypoints = outputs['output_0'][0][0]
  
  for idx, kp in enumerate(keypoints):
    cx = int(kp[1] * img.shape[1])
    cy = int(kp[0] * img.shape[0])
    frame = cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

  cv2.imshow('Video', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

capture.release()
cv2.destroyAllWindows()