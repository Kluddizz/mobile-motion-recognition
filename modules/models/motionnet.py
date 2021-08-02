import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.layers import Dropout, Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU

class MotionNetCNN(tf.keras.Model):
  def __init__(self, classes):
    super().__init__()
    self.conv1 = Conv2D(16, kernel_size=3, strides=2, padding='same', use_bias=False, input_shape=(60, 17, 3,))
    self.bn1 = BatchNormalization()
    self.conv2 = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)
    self.bn2 = BatchNormalization()
    self.conv3 = Conv2D(64, kernel_size=3, strides=2, padding='same', use_bias=False)
    self.bn3 = BatchNormalization()
    self.flatten = Flatten()
    self.fc = Dense(classes, activation=tf.keras.activations.softmax)

  def call(self, x, training=False):
    x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
    x = tf.nn.relu(self.bn2(self.conv2(x), training=training))
    x = tf.nn.relu(self.bn3(self.conv3(x), training=training))
    x = self.flatten(x)
    x = self.fc(x)
    return x

class MotionNet(tf.keras.Model):
  def __init__(self, classes):
    super().__init__()
    self.flatten = Flatten(input_shape=(60, 17, 3,))
    self.dense1 = Dense(128, activation='relu')
    self.dropout = Dropout(0.2)
    self.dense2 = Dense(classes, activation=tf.keras.activations.softmax)

  def call(self, x, training=False):
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.dropout(x)
    x = self.dense2(x)
    return x