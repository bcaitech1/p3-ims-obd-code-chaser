# Swin Transformer for Object Detection
이 저장소는 [mmdetection](https://github.com/open-mmlab/mmdetection) 기반에서 [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf)를 적용하여 Object Detection을 수행 결과를 담기 위한 것이다.

기본 코드는 Microsoft에서 제공한 [official github](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)를 사용했다.

<br>

# Result of HTC
| Backbone | Pretrain | Neck | Lr Schd | Augmentation | box mAP@50|config|
| :---: | :---: | :---: | :---: | :---: | :---: | :--: |
| Swin-T | ImageNet-1K |FPN| 2x | OPTION 1 | 0.392 |[config](./config/htc_swin_T_1.py)|
| Swin-T | ImageNet-1K |FPN| 2x | OPTION 2 | 0.398 |[config](./config/htc_swin_T_2.py)|
| Swin-T | ImageNet-1K |FPN| 2x | OPTION 3 | 0.420 |[config](./config/htc_swin_T_3.py)|
| Swin-S | ImageNet-1K |FPN| 2x | OPTION 1 | 0.393 |[config](./config/htc_swin_S_1.py)|
| Swin-S | ImageNet-1K |FPN| 2x | OPTION 2 | 0.421 |[config](./config/htc_swin_S_2.py)|
| Swin-S | ImageNet-1K |FPN| 2x | OPTION 3 | 0.442 |[config](./config/htc_swin_S_3.py)|

-------------
- **OPTION 1** : RandomFlip, Normalize
- **OPTION 2** : OPTION 1 + ShiftScaleRotate, RandomBrightnessConstrast, MinIoURandomCrop
- **OPTION 3** : OPTION 2 + Instaboost
-------------


<br>

# Usage
## Installation
1. 설치 : Swin Transformer 
   ```shell
   $ cd [PATH]

   $ git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection.git

   $ cd Swin-Transformer-Object-Detection

   $ pip install -e .

   $ cd ../

   $ git clone https://github.com/NVIDIA/apex

   $ cd apex

   $ pip install -v --disable-pip-version-check --no-cache-dir ./
   ```

2. 설치 : 기타 필요 라이브러리
   ```shell
   $ pip install albumentations

   $ pip install instaboostfast
   ```

   만약 설치 후 mmcv 관련 에러가 발생할 경우 아래의 코드를 실행하면 해결되기도 한다.
   ```shell
   $ pip install mmcv-full
   ```

## Train & Inference
> 실행환경은 주피터노트북이지만 아래의 코드가 핵심임을 상기하며 노트북을 보길 바란다.
- Train
  ```python
  config = {config file path}
  cfg = Config.fromfile(config)
  
  model = build_detector(
    cfg.model,
    train_cfg = cfg.get('train_cfg'),
    test_cfg = cfg.get('test_cfg'),
    )
    # pretrained model을 사용하고 싶을 땐 아래 코드의 주석을 해제하여 실행
    # checkpoint = load_checkpoint(model, {checkpoint path}, map_location='cpu')
  datasets = [build_dataset(cfg.data.train)]
  train_detector(model, datasets[0], cfg, distributed=False, validate=True)
  ```
- Inference
  ```python
  config = {config file path}
  cfg = Config.fromfile(config)

  dataset = build_dataset(cfg.data.test)
  data_loader = build_dataloader(
      dataset,
      samples_per_gpu=1,
      workers_per_gpu=cfg.data.workers_per_gpu,
      dist=False,
      shuffle=False
  )

  model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
  checkpoint = load_checkpoint(model, {checkpoint path}, map_location='cpu')
  model = MMDataParallel(model.cuda(), device_ids=[0])
  
  output = single_gpu_test(model, data_loader, show_score_thr=0.05)
  ```