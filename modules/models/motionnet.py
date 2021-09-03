import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.python.keras.backend import relu

class MotionNetCNN(tf.keras.Model):
  def __init__(self, classes, blocks=[16, 32, 64]):
    super().__init__()
    self.flatten = Flatten()
    self.fc = Dense(classes, activation=tf.keras.activations.softmax)

    self.blocks = tf.keras.Sequential()

    for i, block_channels in enumerate(blocks):
      if i == 0:
        conv = Conv2D(block_channels, kernel_size=3, strides=2, padding='same', use_bias=False, input_shape=(10, 17, 3,))
      else:
        conv = Conv2D(block_channels, kernel_size=3, strides=2, padding='same', use_bias=False)

      self.blocks.add(conv)
      self.blocks.add(BatchNormalization())
      self.blocks.add(tf.keras.layers.ReLU())


  def call(self, x, training=False):
    x = self.blocks(x, training=training)
    x = self.flatten(x)
    x = self.fc(x)
    return x

class MotionNet(tf.keras.Model):
  def __init__(self, classes, blocks=[128]):
    super().__init__()
    self.flatten = Flatten(input_shape=(60, 17, 3,))
    self.dense1 = Dense(128, activation='relu')
    self.dropout = Dropout(0.2)
    self.dense2 = Dense(classes, activation=tf.keras.activations.softmax)

    self.blocks = tf.keras.Sequential()

    for block_neurons in blocks:
      self.blocks.add(Dense(block_neurons, activation='relu'))
      self.blocks.add(Dropout(0.2))

  def call(self, x, training=False):
    x = self.flatten(x)
    x = self.blocks(x)
    x = self.dense2(x)
    return x