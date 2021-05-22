# TEAM : Code Chaser
- member
  - 반영성(https://github.com/ys-ban)
  - 소재열
  - 이대훈
  - 조호성(https://github.com/chohoseong)
  - 최이서

# Task and solving strategies
- data: a part of [TACO dataset](http://tacodataset.org/)
- problem and breakthrough:
  - overfitting -> multilabel stratified 5-fold cross validation, lr scheduling, weight decay
  - inconsistency of images -> augmentations(flip, grid mask, rotate, etc)
- semantic segmentation
  - **architecture**: UNet3+ with depp supervision and class guide module
  - **backbone**: EfficientNet-B3(pretrained, noisy student)
  - **validation**: multilabel stratified 5-fold cross validation
  - **ensemble**: 5-fold soft voting
- object detection
  - model1:
    - **backbone**: Swin Transformer
    - **neck**: FPN
    - **detector**: Hybrid Task Cascade
  - model2:
    - **backbone**: EfficientNet-B5
    - **neck**: BiFPN
    - **detector**: EfficientDet
  - model3:
    - **backbone**: 
    - **neck**: 
    - **detector**: 
  - multi head ensemble based on [WBF](https://arxiv.org/abs/1910.13302)

# Result
- semantic segmentation
  - mIoU - 0.6795
- object detection
  - mAP@50 - 0.4171