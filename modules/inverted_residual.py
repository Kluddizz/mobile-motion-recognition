# This code is adapted from Luis Gonzales (Medium)
#
# Article: https://medium.com/@luis_gonzales/a-look-at-mobilenetv2-inverted-residuals-and-linear-bottlenecks-d49f85c12423
# Source: https://gist.github.com/luis-gonzales/464480f08e3df7a4bd98a0734b266737

import tensorflow as tf

class InvertedResidual(tf.keras.layers.Layer):
  def __init__(self, filters, strides, expansion_factor=6, trainable=True, name=None, **kwargs):
    super(InvertedResidual, self).__init__(trainable=trainable, name=name, **kwargs)
    self.filters = filters
    self.strides = strides
    self.expansion_factor = expansion_factor

  def build(self, input_shape):
    input_channels = int(input_shape[3])
    self.conv1 = tf.keras.layers.Conv2D(
      filters=int(input_channels*self.expansion_factor),
      kernel_size=1,
      use_bias=False,
    )
    self.bn1 = tf.keras.layers.BatchNormalization()

    self.dwise1 = tf.keras.layers.DepthwiseConv2D(
      kernel_size=3,
      strides=self.strides,
      padding="same",
      use_bias=False,
    )
    self.bn2 = tf.keras.layers.BatchNormalization()

    self.conv2 = tf.keras.layers.Conv2D(
      filters=self.filters,
      kernel_size=1,
      use_bias=False,
    )
    self.bn3 = tf.keras.layers.BatchNormalization()

  def call(self, input_x):
    x = self.conv1(input_x)
    x = self.bn1(x)
    x = tf.nn.relu6(x)

    x = self.dwise1(x)
    x = self.bn2(x)
    x = tf.nn.relu6(x)

    x = self.conv2(x)
    x = self.bn3(x)

    if input_x.shape[1:] == x.shape[1:]:
      x += input_x

    return x

  def get_config(self):
    cfg = super(InvertedResidual, self).get_config()
    cfg.update({
      'filters': self.filters,
      'strides': self.strides,
      'expansion_factor': self.expansion_factor,
    })