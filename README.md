# Pose estimation and motion analysis for mobile devices.
This repository provides implementations of diverse pose estimation and motion
analysis models especially for the use on mobile devices using [PyTorch](https://pytorch.org/).

## Helpful links
* [A Look At MobileNetV2 Inverted Residuals And Linear Bottlenecks](https://medium.com/@luis_gonzales/a-look-at-mobilenetv2-inverted-residuals-and-linear-bottlenecks-d49f85c12423)
* [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
* [Next-Generation Pose Detection with MoveNet and TensorFlow.js](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html)

## Benchmarks
**AMD Ryzen 7 5800X @ 3.8GHz**

    model                               ms    fps
    -------------------------------  -----  -----
    mobilenet_v2_fpn_centernet        26.19     38
    mobilenet_v3_large_fpn_centernet  18.35     55
    mobilenet_v3_small_fpn_centernet   9.85    102

**MSI RTX 3080 Suprim X 10G LHR**

    model                              ms    fps
    -------------------------------  ----  -----
    mobilenet_v2_fpn_centernet        5.76    173
    mobilenet_v3_large_fpn_centernet  6.66    150
    mobilenet_v3_small_fpn_centernet  5.74    174

## Training
To train your own model just run the training script.

    python train.py --help

## Evaluation
To evaluate trained models just create a new folder `./eval` and place your images inside it.

    mkdir eval
    cp /path/to/images/* ./eval

Now run the evaluation script `eval.py`.

    python eval.py