# Video captioning models in Pytorch (Work in progress)
This repository contains Pytorch implementation of video captioning SOTA models from 2015-2019 on MSVD and MSRVTT datasets. Details are given in below table

 | Model | Datasets | Paper name | Year | Status |
 | :---: | :---: | :---: | :---: |  :---: |
 | Mean Pooling | MSVD, MSRVTT | Translating videos to natural language using deep recurrent neural networks[[1]](#1) | 2015 | Implemented |
 | S2VT | MSVD, MSRVTT | Sequence to Sequence - Video to Text[[2]](#2) | 2015 | Implemented |
 | SA-LSTM | MSVD, MSRVTT | Describing videos by exploiting temporal structure[[3]](#3) | 2015 | Implemented |
 | Recnet | MSVD, MSRVTT |  Reconstruction Network for Video Captioning[[4]](#4)  | 2018 | Implemented |
 | MARN | MSVD, MSRVTT | Memory-Attended Recurrent Network for Video Captioning[[5]](#5) | 2019 | Implemented |
 
 *More recent models will be added in future

## Environment
* Ubuntu 18.04
* CUDA 11.0
* Nvidia GeForce RTX 2080Ti

## Requirements 
* Java 8 
* Python 3.8.5
    * Pytorch 1.7.0
    * Other Python libraries specified in requirements.txt

## How to Use
 
### Step 1. Setup python virtual environment

```
$ virtualenv .env
$ source .env/bin/activate
(.env) $ pip install --upgrade pip
(.env) $ pip install -r requirements.txt
```
### Step 2. Prepare data, path and hyperparameter settings
1. Extract features from network you want to use, and locate them at `<PROJECT ROOT>/<DATASET>/features/<DATASET>_APPEARANCE_<NETWORK>_<FRAME_LENGTH>.hdf5`. To extracted features follow the repository [here](https://github.com/nasib104/video_feature_extraction). Or simply download the already extracted features from given table and locate them in `<PROJECT ROOT>/<DATASET>/features/`

   | Dataset | Feature Type | Inception-v4 | InceptionResNetV2 | ResNet-101 | REsNext-101 |
   | :---: | :---: |  :---: | :---: | :---: | :---: |
   | MSVD | Appearance | [link](https://www.dropbox.com/s/m8llhpvxpzge5jj/MSVD_APPEARANCE_INCEPTIONV4_28.hdf5?dl=0) | [link](https://www.dropbox.com/s/1podxw82gl1pavg/MSVD_APPEARANCE_INCEPTIONRESNETV2_28.hdf5?dl=0) | [link](https://www.dropbox.com/s/34ay6if4j2gfcgz/MSVD_APPEARANCE_RESNET101_28.hdf5?dl=0) | - |
   | MSR-VTT | Appearance | [link](https://www.dropbox.com/s/13k4rruu84a42va/MSRVTT_APPEARANCE_INCEPTIONV4_28.hdf5?dl=0) | [link](https://www.dropbox.com/s/j15hkyw4sy59cxp/MSRVTT_APPEARANCE_INCEPTIONRESNETV2_28.hdf5?dl=0) | [link](https://www.dropbox.com/s/yfwjps6cs0y8drm/MSRVTT_APPEARANCE_RESNET101_28.hdf5?dl=0) | - |
   | MSVD | Motion | - | - | - | [link](https://www.dropbox.com/s/1m7leypc6xgmb35/MSVD_MOTION_RESNEXT101.hdf5?dl=0) |

You can change hyperparameters by modifying `config.py`.

### Step 3. Prepare Evaluation Codes
Clone evaluation codes from [the official coco-evaluation repo](https://github.com/tylin/coco-caption).

   ```
   (.env) $ git clone https://github.com/tylin/coco-caption.git
   (.env) $ mv coco-caption/pycocoevalcap .
   (.env) $ rm -rf coco-caption
   ```
   Or simply copy the pycocoevalcap folder and its contents in the project root.

### Step 4. Training
Follow the demo given in `video_captioning.ipynb`.

### Step 5. Inference
Follow the demo given in `video_captioning.ipynb`.

## Quantitative Results

*MSVD
 | Model | Pretrained model | BLEU4 | METEOR | ROUGE_L | CIDER | Pretrained |
 | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | Mean Pooling | Inceptionv4 | 42.2 | 31.6 | 68.2 | 69.7 | [link](https://drive.google.com/file/d/1Oj5yMIKnU5obe0UXTknvX9S70CFVH7nz/view?usp=sharing) |
 | SA-LSTM | InceptionvResNetV2 | 45.5 | 32.5 | 69.0 | 78.0 | [link](https://www.dropbox.com/s/bs6mepcv8oucnfb/sa_lstm_msvd.pt?dl=0) |
 | S2VT | Inceptionv4 | - | - | - | - | - |
 | RecNet (global ) | Inceptionv4 | - | - | - | - | - |
 | RecNet (local) | Inceptionv4 | -| - | - | - | - |
 | MARN | Inceptionv4 | - | - | - | - | - |
 
 *MSRVTT
 
 | Model | Pretrained model | BLEU4 | METEOR | ROUGE_L | CIDER | Pretrained |
 | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | Mean Pooling | Inceptionv4 | 34.9 | 25.5 | 58.12 | 35.76 | [link](https://drive.google.com/file/d/1YhBkQnR4MXWhmHRufXmNULSDzWhJdbL3/view?usp=sharing) |
 | SA-LSTM | Inceptionv4 | - | - | - | - | - |
 | S2VT | Inceptionv4 | - | - | - | - | - |
 | RecNet (global ) | Inceptionv4 | - | - | - | - | - |
 | RecNet (local) | Inceptionv4 | -| - | - | - | - |
 | MARN | Inceptionv4 | - | - | - | - | - |

# References
<a id="1">[1]</a>
S. Venugopalan, H. Xu, J. Donahue, M. Rohrbach,
R. Mooney, and K. Saenko. Translating videos to natural
language using deep recurrent neural networks. In NAACLHLT, 2015.

<a id = "2">[2]</a>
Subhashini Venugopalan, Marcus Rohrbach, Jeff Donahue, Raymond J. Mooney, 
Trevor Darrell and Kate Saenko. Sequence to Sequence - Video to Text. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2015

<a id = "3">[3]</a>
Yao, Li, et al. "Describing videos by exploiting temporal structure." Proceedings of the IEEE international conference on computer vision. 2015.

<a id = "4">[4]</a>
Wang, Bairui, et al. "Reconstruction Network for Video Captioning." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

<a id = "5">[5]</a>
Wenjie Pei, Jiyuan Zhang, Xiangrong Wang, Lei Ke, Xiaoyong Shen, and Yu-Wing Tai. Memory-attended recurrent network for video captioning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8347–8356, 2019

## Acknowlegement

I got some of the coding ideas and the extracted video features from
[hobincar/pytorch-video-feature-extractor](https://github.com/hobincar/pytorch-video-feature-extractor)
Many thanks!
