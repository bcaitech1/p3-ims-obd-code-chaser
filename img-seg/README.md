# UNet3+ for Semantic Segmentation
이 저장소는 [‘UNet 3+: A full-scale connected unet for medical image segmentation’](https://arxiv.org/abs/2004.08790)을 구현한 코드이다.

기본 코드는 공식 [github](https://github.com/ZJUGiveLab/UNet-Version)의 코드를 사용했다.

기존 공개된 코드는 의료이미지 분류 문제에 기반하여 만들어졌기에 binary segmentation만을 수행할 수 있었고 해당 부분을 다중 분류가 가능하도록 수정했으며 동시에 그에 맞게 CGM 코드 또한 수정했다.



<br>

# Result
| Backbone | Pretrain | Depth | mIoU|
| :---: | :---: | :---: | :---: | 
| EfficientNet b3 | noisy-student | 1 | 0.5904 |
| EfficientNet b3 | noisy-student | 2 | 0.6119 |
| EfficientNet b3 | noisy-student | 3 | 0.6221 |
| EfficientNet b3 | noisy-student | 4 | 0.6153 |
| EfficientNet b3 | noisy-student | 1+2+3 | 0.6511 |
| EfficientNet b3 | noisy-student | 1+2+3+4 | 0.6767 |
| EfficientNet b3 | noisy-student | 5 fold ensemble  | 0.6933 |

-------------
- **Model** : UNet3+ with deep supervision and class guide module
- **Augmentation** : RandomResizedCrop, RandomBrightnessContrast, ShiftScaleRotate, RandomFlip
- **Lr schedule** : CosineAnnealing with lr=3e-4, min_lr=1e-6, T=10
-------------


<br>

# Usage
## Train & Inference
- Train
  ```shell
  python train.py \
            --seed {seed} \
            --epochs {epochs} \
            --batch_size {batch_size} \
            --lr {lr} \
            --name {check save folder name} \
            --data_dir {data_dir} \
            --model_dir {model_dir}
  ```
- Inference
  ```shell
  python inference.py \
            --batch_size {batch_size} \
            --data_dir {data_dir} \
            --model_dir {model_dir}
  ```