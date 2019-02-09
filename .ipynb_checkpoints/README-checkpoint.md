## SSD: Single-Shot MultiBox Detector implementation in Keras for pedestrian detection
---
### Contents

1. [Overview](#overview)
2. [Performance](#performance)
3. [Examples](#examples)
4. [Dependencies](#dependencies)
5. [How to use it](#how-to-use-it)
6. [Download the convolutionalized VGG-16 weights](#download-the-convolutionalized-vgg-16-weights)
7. [Download the original trained model weights](#download-the-original-trained-model-weights)
8. [How to fine-tune one of the trained models on your own dataset](#how-to-fine-tune-one-of-the-trained-models-on-your-own-dataset)
9. [ToDo](#todo)
10. [Important notes](#important-notes)
11. [Terminology](#terminology)

### Overview

This is a Keras port of the SSD model architecture introduced by Wei Liu et al. in the paper [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325).

Ports of the trained weights of all the original models are provided below. This implementation is accurate, meaning that both the ported weights and models trained from scratch produce the same mAP values as the respective models of the original Caffe implementation (see performance section below).

The main goal of this project is to create an SSD implementation that is well documented for those who are interested in a low-level understanding of the model. The provided tutorials, documentation and detailed comments hopefully make it a bit easier to dig into the code and adapt or build upon the model than with most other implementations out there (Keras or otherwise) that provide little to no documentation and comments.

The repository currently provides the following network architectures:
* SSD300: [`keras_ssd300.py`](models/keras_ssd300.py)
* SSD512: [`keras_ssd512.py`](models/keras_ssd512.py)
* SSD7: [`keras_ssd7.py`](models/keras_ssd7.py) - a smaller 7-layer version that can be trained from scratch relatively quickly even on a mid-tier GPU, yet is capable enough for less complex object detection tasks and testing. You're obviously not going to get state-of-the-art results with that one, but it's fast.

If you would like to use one of the provided trained models for transfer learning (i.e. fine-tune one of the trained models on your own dataset), there is a [Jupyter notebook tutorial](weight_sampling_tutorial.ipynb) that helps you sub-sample the trained weights so that they are compatible with your dataset, see further below.

If you would like to build an SSD with your own base network architecture, you can use [`keras_ssd7.py`](models/keras_ssd7.py) as a template, it provides documentation and comments to help you.

### Performance

### Examples

### Dependencies

* Python 3.x
* Numpy
* TensorFlow 1.x
* Keras 2.x
* OpenCV
* Beautiful Soup 4.x
