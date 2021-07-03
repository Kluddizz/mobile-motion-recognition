import tensorflow as tf
from inverted_residual import InvertedResidual

class FPN(tf.keras.Model):
  def __init__(self):
    super(FPN, self).__init__()
    self.in_channels = 32

    self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding="same", use_bias=False)
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.relu1 = tf.keras.layers.ReLU(6.0)

    self.bottleneck1 = self._make_layer( 16, 1, stride=1, expansion_factor=1)
    self.bottleneck2 = self._make_layer( 24, 2, stride=2, expansion_factor=6)
    self.bottleneck3 = self._make_layer( 32, 3, stride=2, expansion_factor=6)
    self.bottleneck4 = self._make_layer( 64, 4, stride=2, expansion_factor=6)
    self.bottleneck5 = self._make_layer( 96, 3, stride=1, expansion_factor=6)
    self.bottleneck6 = self._make_layer(160, 3, stride=2, expansion_factor=6)
    self.bottleneck7 = self._make_layer(320, 1, stride=1, expansion_factor=6)

    self.conv2 = tf.keras.layers.Conv2D(1280, kernel_size=1, strides=1, padding="same", use_bias=False)
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.relu2 = tf.keras.layers.ReLU(6.0)

    # Smooth layers
    self.smooth1 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
    self.smooth2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
    self.smooth3 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
    self.smooth4 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")

    # Lateral layers
    self.lateral1 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="valid")
    self.lateral2 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="valid")
    self.lateral3 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="valid")
    self.lateral4 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="valid")

  def _make_layer(self, channels, num_blocks, strides, expansion_factor):
    layers = []

    for i in range(0, num_blocks):
      layers.append(InvertedResidual(channels, strides=strides, expansion_factor=expansion_factor))

    return tf.keras.Sequential(layers)

  def _upsample_add(self, x, y):
    _, H, W, C = y.shape
    return tf.image.resize(x, size=(H, W), method="bilinear")

  def call(self, x, training=False):
    c1 = self.relu1(self.bn1(self.conv1(x), training=training))
    