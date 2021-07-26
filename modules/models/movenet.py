import tensorflow as tf
from tensorflow.keras.layers import ReLU, Dropout, Dense, Flatten

class MotionNet(tf.keras.Model):
  def __init__(self, classes):
    super().__init__(classes)
    self.classes = classes

    self.head = tf.keras.Sequential()
    self.head.add(Flatten(input_shape=(60, 17, 3,)))
    print(self.head.input_shape, self.head.output_shape)
    self.head.add(Dense(128, activation='relu'))
    print(self.head.output_shape)
    self.head.add(Dropout(0.2))
    print(self.head.output_shape)
    self.head.add(Dense(classes, activation=tf.keras.activations.softmax))
    print(self.head.output_shape)

  def call(self, x, training=False):
    x = self.head(x, training=training)
    return x