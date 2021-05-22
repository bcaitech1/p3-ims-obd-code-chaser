# EfficientDet for Object Detection

이 저장소는 [EfficientDet](https://arxiv.org/pdf/1905.11946.pdf)를 적용하여 Object Detection을 수행 결과를 담기 위한 것이다.

기본 코드는 [github](https://github.com/rwightman/efficientdet-pytorch)를 사용했다.

<br>

# Usage

## Installation

1. 설치 : 필요 라이브러리

   ```shell
   $ pip install albumentations==0.5.2

   $ pip install efficientnet_pytorch

   $ pip install pycocotools

   $ pip install cython

   $ pip install timm

   $ pip install effdet

   $ pip install gluoncv

   $ pip install mxnet
   ```

## Train & Inference

> 실행환경은 주피터노트북과 파이썬 파일 두 가지 형식으로 제공한다.

- Train

  ```shell
   $ python train.py \
            --epochs {epochs} \
            --train_batch_size {batch_size} \
            --valid_batch_size {batch_size} \
            --model {model} \
            --optimizer {optimizer} \
            --lr {lr}
  ```

  <br>

- Inference

  ```shell
   $ python inference.py
            --batch_size {batch_size} \
            --threshold {threshold} \
            --check_dir {save folder name}
  ```
