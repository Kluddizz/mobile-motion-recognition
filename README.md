# Pose estimation and motion analysis for mobile devices.
This repository provides implementations of diverse pose estimation and motion
analysis models especially for the use on mobile devices using [PyTorch](https://pytorch.org/).

## Helpful links
* [A Look At MobileNetV2 Inverted Residuals And Linear Bottlenecks](https://medium.com/@luis_gonzales/a-look-at-mobilenetv2-inverted-residuals-and-linear-bottlenecks-d49f85c12423)
* [Next-Generation Pose Detection with MoveNet and TensorFlow.js](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html)

## Training
To train your own model just run the training script.

    python train.py --help

## Evaluation
To evaluate trained models just create a new folder `./eval` and place your images inside it.

    mkdir eval
    cp /path/to/images/* ./eval

Now run the evaluation script `eval.py`.

    python eval.py