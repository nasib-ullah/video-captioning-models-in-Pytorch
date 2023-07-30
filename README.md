# Video captioning models in Pytorch (Work in progress)
[![DOI](https://zenodo.org/badge/333657270.svg)](https://zenodo.org/badge/latestdoi/333657270)

This repository contains Pytorch implementation of video captioning SOTA models from 2015-2020 on MSVD and MSRVTT datasets. Details are given in below table

 | Model | Datasets | Paper name | Year | Status | Remarks |
 | :---: | :---: | :---: | :---: |  :---: | :---: |
 | Mean Pooling | MSVD, MSRVTT | Translating videos to natural language using deep recurrent neural networks[[1]](#1) | 2015 | Implemented | No temporal modeling|
 | S2VT | MSVD, MSRVTT | Sequence to Sequence - Video to Text[[2]](#2) | 2015 | Implemented | Single LSTM as encoder decoder |
 | SA-LSTM | MSVD, MSRVTT | Describing videos by exploiting temporal structure[[3]](#3) | 2015 | Implemented | Good Baseline with attention | 
 | Recnet | MSVD, MSRVTT |  Reconstruction Network for Video Captioning[[4]](#4)  | 2018 | Implemented | Results did not improve over SA-LSTM with both global and local reconstruction loss |
 | MARN | MSVD, MSRVTT | Memory-Attended Recurrent Network for Video Captioning[[5]](#5) | 2019 | Implemented | Memory requirement linearly increases with vocabulary size |
 | ORG-TRL| MSVD, MSRVTT | Object Relational Graph with Teacher-Recommended Learning for Video Captioning[[6]](#6) | 2020 | In progress | leavarage GCN for object relational features |
 
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
   | MSVD | Appearance | [link](https://www.dropbox.com/s/mtrb0t1phfjdr9u/MSVD_APPEARANCE_INCEPTIONV4.hdf5?dl=0) | [link](https://www.dropbox.com/s/zy68cormb8xhfw2/MSVD_APPEARANCE_INCEPTIONRESNETV2_28.hdf5?dl=0) | [link](https://www.dropbox.com/s/yd2rua9k0v5iigs/MSVD_APPEARANCE_RESNET101_28.hdf5?dl=0) | - |
   | MSR-VTT | Appearance | [link](https://www.dropbox.com/s/8msnmsaoq0739k4/MSRVTT_APPEARANCE_INCEPTIONV4_28.hdf5?dl=0) | [link](https://www.dropbox.com/s/t4pfa2kwe42jay8/MSRVTT_APPEARANCE_INCEPTIONRESNETV2_28.hdf5?dl=0) | [link](https://www.dropbox.com/s/x3tb68q6xa6qi28/MSRVTT_APPEARANCE_RESNET101_28.hdf5?dl=0) | - |
   | MSVD | Motion | - | - | - | [link](https://www.dropbox.com/s/8rril55dj06oxmx/MSVD_MOTION_RESNEXT101.hdf5?dl=0) |
   | MSR-VTT | Motion | - | - | - | [link](https://www.dropbox.com/s/8mf9jzsfopekogr/MSRVTT_MOTION_RESNEXT.hdf5?dl=0) |
   | MSVD | Object | - | - | [link](https://www.dropbox.com/s/5xlefcuoh3j2fq4/MSVD_OBJECT_FASTERRCNN_R101FC2_28.hdf5?dl=0) | - |
   | MSRVTT | Object | - | - | [link](https://www.dropbox.com/s/5ace1fve3yusqox/MSRVTT_OBJECT_FASTERRCNN_R101FC2_28.hdf5?dl=0) | - |

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
 | Mean Pooling | Inceptionv4 | 42.4 | 31.6 | 68.3 | 71.8 | [link](https://www.dropbox.com/s/4rf0zw2csjg1rxs/mp_lstm_msvd.pt?dl=0) |
 | SA-LSTM | InceptionvResNetV2 | 45.5 | 32.5 | 69.0 | 78.0 | [link](https://www.dropbox.com/s/yoaegk4hglyg79l/sa_lstm_msvd.pt?dl=0) |
 | S2VT | Inceptionv4 | - | - | - | - | - |
 | RecNet (global ) | Inceptionv4 | - | - | - | - | - |
 | RecNet (local) | Inceptionv4 | -| - | - | - | - |
 | MARN | Inceptionv4, REsNext-101 | 48.5 | 34.4 | 71.4 | 86.4 | [link](https://www.dropbox.com/s/9ryeg03bd8mtm8s/marn_msvd.pt?dl=0) |
 |ORG-TRL| InceptionResNetV2, REsNext-101 | - | - | - | - | 
 
 *MSRVTT
 
 | Model | Pretrained model | BLEU4 | METEOR | ROUGE_L | CIDER | Pretrained |
 | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | Mean Pooling | Inceptionv4 | 34.9 | 25.5 | 58.12 | 35.76 | [link](https://drive.google.com/file/d/1YhBkQnR4MXWhmHRufXmNULSDzWhJdbL3/view?usp=sharing) |
 | SA-LSTM | Inceptionv4 | - | - | - | - | - |
 | S2VT | Inceptionv4 | - | - | - | - | - |
 | RecNet (global ) | Inceptionv4 | - | - | - | - | - |
 | RecNet (local) | Inceptionv4 | -| - | - | - | - |
 | MARN | Inceptionv4 | - | - | - | - | - |
 |ORG-TRL| InceptionResNetV2, REsNext-101 | - | - | - | - | 

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

<a id = "6">[6]</a>
Ziqi Zhang, Yaya Shi, Chunfeng Yuan, Bing Li, Peijin Wang, Weiming Hu, Zhengjun Zha. Object Relational Graph with Teacher-Recommended Learning for Video Captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.

## Acknowlegement

I got some of the coding ideas from
[hobincar/pytorch-video-feature-extractor](https://github.com/hobincar/pytorch-video-feature-extractor). For pre-trained appearance feature extraction I have followed [this repo](https://github.com/Cadene/pretrained-models.pytorch) and [this repo](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) for 3D motion feature extraction. Many thanks!
