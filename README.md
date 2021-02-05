# Video captioning models in Pytorch
This repository contains Pytorch implementation of video captioning SOTA models from 2015-2019 on MSVD and MSRVTT datasets. Details are given in below table

 | Model | Datasets | Paper name | Year | 
 | :---: | :---: | :---: | :---: | 
 | Mean Pooling | MSVD, MSRVTT | Translating videos to natural language using deep recurrent neural networks[[1]](#1) | 2015 | 
 | S2VT | MSVD, MSRVTT | Sequence to Sequence - Video to Text[[2]](#2) | 2015 |
 | SA-LSTM | MSVD, MSRVTT | - | 2016 |
 | Recnet | MSVD, MSRVTT |  Reconstruction Network for Video Captioning[[3]](#3)  | 2018 |
 | MARN | MSVD, MSRVTT | Memory-Attended Recurrent Network for Video Captioning[[4]](#4) | 2019 |
 
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
1. Extract features from network you want to use, and locate them at `<PROJECT ROOT>/<DATASET>/features/<PRETRAINED_NETWORK_NAME>/<DATASET>_<NETWORK>.hdf5`. I extracted features of Inception-v4 from [here](https://github.com/hobincar/pytorch-video-feature-extractor)

   | Dataset | Inception-v4 |
   | :---: | :---: | 
   | MSVD | [link](https://drive.google.com/open?id=18aZ8AdFeJ8h2wPR3YMnZNHnw7ebtfGih) |
   | MSR-VTT | [link](https://drive.google.com/open?id=1pFh4u-KwSnCFRl6UJgg7yeaLo2GbxkVT) |

You can change hyperparameters by modifying `config.py`.

### Step 3. Prepare Evaluation Codes
Clone evaluation codes from [the official coco-evaluation repo](https://github.com/tylin/coco-caption).

   ```
   (.env) $ git clone https://github.com/tylin/coco-caption.git
   (.env) $ mv coco-caption/pycocoevalcap .
   (.env) $ rm -rf coco-caption
   ```

### Step 4. Training
Follow the demo given in `video_captioning.ipynb`.

### Step 5. Inference
Follow the demo given in `video_captioning.ipynb`.

## Quantitative Results

*MSVD
 | Model | Pretrained model | BLEU4 | METEOR | CIDER | ROUGE_L | Pretrained |
 | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | Mean Pooling | DenseNet | - | - | - | - | - |
 | Mean Pooling | Inceptionv4 | - | - | - | - | - |
 | SA-LSTM | Inceptionv4 | - | - | - | - | - |
 | S2VT | Inceptionv4 | - | - | - | - | - |
 | RecNet (global ) | Inceptionv4 | - | - | - | - | - |
 | RecNet (local) | Inceptionv4 | -| - | - | - | - |
 | MARN | Inceptionv4 | - | - | - | - | - |
 
 *MSRVTT
 
 | Model | Pretrained model | BLEU4 | METEOR | CIDER | ROUGE_L | Pretrained |
 | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | Mean Pooling | Inceptionv4 | 34.9 | 25.5 | 35.76 | 58.12 | [link](https://drive.google.com/file/d/1YhBkQnR4MXWhmHRufXmNULSDzWhJdbL3/view?usp=sharing) |
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
Wang, Bairui, et al. "Reconstruction Network for Video Captioning." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

<a id = "4">[4]</a>
Wenjie Pei, Jiyuan Zhang, Xiangrong Wang, Lei Ke, Xiaoyong Shen, and Yu-Wing Tai. Memory-attended recurrent network for video captioning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8347–8356, 2019

## Acknowlegement

I got some of the coding idea and the extracted video features from
[hobincar/pytorch-video-feature-extractor](https://github.com/hobincar/pytorch-video-feature-extractor)
Many thanks!
