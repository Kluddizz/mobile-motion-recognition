import argparse
import tensorflow as tf

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('saved_model_dir', type=str)
  parser.add_argument('--out', '-o', type=str)
  args = parser.parse_args()

  converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model_dir)
  tflite_model = converter.convert()

  output_filename = args.out or 'converted.tflite'

  with open(output_filename, 'wb') as f:
    f.write(tflite_model)