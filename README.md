## SSD: Single-Shot MultiBox Detector implementation in Keras for pedestrian detection
---
### Contents

1. [Overview](#overview)
2. [Performance](#performance)
3. [Pascal VOC](#pascal)
4. [Dependencies](#dependencies)
5. [Examples](#examples)
6. [Weights](#weights)

### Overview

This is a Keras port of the SSD model architecture introduced by Wei Liu et al. in the paper [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325).

This repository provides an extension from https://github.com/pierluigiferrari/ssd_keras adding new features stractors and a new network architecture for detection. All architectures where trained for pedestrian detection using PASCAL VOC 2007 and 2012.
Currently this implementation contains the following feature stractors:
- Mobilenetv1
- Mobilenetv2
- Shufflenetv1
- Shufflenetv2
- Squeezenet

SSD/NoShuffleSE uses a reduce version of the Shufflenetv2 architecture and Squeeze and Excitations units.
The reduce versión of the Shufflenetv2 architecture is shown below.
The Squeeze and Excitations units are applied after a 1x1 Conv.

### Evaluation
##### IOU
$$IOU(B_{1,}B_{2})=\frac{Area(B_{1}\cap B_{2})}{Area(B_{1}\cup B_{2})}$$

Where $$B_{1}$$ and $$B_{2}$$ are bounding boxes
##### Precision
$$P=\frac{{\textstyle \sum_{k=1}^{N}\sum_{j=1}^{\mid D_{k}\mid}BestMatch(G_{k},D_{k,j})}}{\sum_{k=1}^{N}\mid D_{k}\mid}$$

Where:
N is the number of images.
k is the image list index.
$$D_{k}$$ is the prediction list of image k.
$$G_{k}$$ is the ground truth of image k.
i and j are list indexes

$$BestMatch(G_{k},D_{k,j})=\left\{ \begin{matrix}0 & if\,\,\,\,\,\,\,max_{i}IOU(G_{k,i},D_{k,j})\leq0.5\\
1 & if\,\,\,\,\,\,\,max_{i}IOU(G_{k,i},D_{k,j})>0.5
\end{matrix}\right.$$
##### Recall
$$R=\frac{\sum_{k=1}^{N}\sum_{j=1}^{\mid G_{k}\mid}BestMatch(G_{k,j},D_{k})}{\sum_{k=1}^{N}\mid G_{k}\mid}$$

$$BestMatch(G_{k,j},D_{k})=\left\{ \begin{matrix}0 & s\acute{\imath}\,\,\,\,\,\,\,max_{i}IOU(G_{k,j},D_{k,i})\leq0.5\\
1 & s\acute{\imath}\,\,\,\,\,\,\,max_{i}IOU(G_{k,j},D_{k,i})>0.5
\end{matrix}\right.$$
##### F1 score
$$F_{1}=\frac{2PR}{P+R}$$
### Performance
| Arquitectura      | Training set | F1 score | FPS |
|-------------------|--------------|----------|-----|
| SSD/Mobilenetv1   | 07++12       | 0.6501   | **22**  |
| SSD/Mobilenetv2   | 07++12       | 0.6094   | 12  |
| SSD/Shufflenetv1  | 07++12       | 0.6121   | 12  |
| SSD/Shufflenetv2  | 07++12       | 0.6743   | 14  |
| SSD/Squeezenet SB | 07++12       | 0.5722   | 35  |
| SSD/NoShuffleSE   | 07++12       | 0.6768   | 20  |
| SSD/NoShuffleSE   | 07++12+COCO  | **0.7094**   | 20  |

SB indicates simple bypass
The time is measure with Intel® Core™ i7-7740X CPU @ 4.30GHz × 8
SSD/NoShuffleSE achieves an Average Precision of **66.0** on PASCAL VOC 2007

### PASCAL 
You can download the PASCAL dataset in the format needed for the model in [download_pascal_dataset.ipynb](download_pascal_dataset.ipynb)

### Dependencies

* Python 3.x
* Numpy
* TensorFlow 1.x
* Keras 2.x
* OpenCV
* Beautiful Soup 4.x
