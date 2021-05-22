# Trash object detection using YOLOv5
이 저장소는 [YOLO v5](https://github.com/ultralytics/yolov5) 를 적용하여 trash Object Detection을 수행한 결과를 담기 위한 것이다.


<br>

# Results
| Model | Evolved hyperparmeter | box mAP@50|config|Wandb|
| :---: | :---: | :---: | :---: | :---: |
| YOLOv5x6 |  OPTION 1 | 0.4184 |[config](./config/hyp_evolved.yaml)|[wabdb](https://wandb.ai/erinchoi/mixtest/reports/Training-with-hyp_evolved-yaml--Vmlldzo3MTQ0NDI?accessToken=pg598rh5dfom1feoanvlu1zpppjaimoqn1fibns5wqerhsxcqbdskxl09s43cgyn)|
| YOLOv5x6 |  OPTION 2 | 0.4310 |[config](./config/htc_mixup.yaml)|[wabdb](https://wandb.ai/erinchoi/Kfold/reports/Training-with-hyp_mixup-yaml--Vmlldzo3MTQ0NDY?accessToken=o4k067wo9qzb6q42vx1ta3df0ddfueqe1t3c5ui5eh09284o2dpb7bmwr07ef5ai)|
| YOLOv5x6 kfold |  OPTION 1 | 0.5001 |[config](./config/hyp_evolved.yaml)|NONE
| YOLOv5x6 kfold TTA |  OPTION 1 | 0.4993 |[config](./config/hyp_evolved.yaml)|NONE
| YOLOv5x6 kfold |  OPTION 2 | 0.5076 |[config](./config/htc_mixup.yaml)|NONE


-------------
- **OPTION 1** : 196th evolved hyper-parameters without mix-up augmentation
- **OPTION 2** : 57th evolved hyper-parameters with mix-up augmentation
-------------


<br>

# Usage
## Installation
1. 설치 : YOLO v5 
   ```shell
   $ cd [PATH]

   $ git clone https://github.com/ultralytics/yolov5.git

   $ cd yolov5

   $ pip install -r requirements.txt

   ```

2. Convert2YOLO
   ```shell
   $ git clone https://github.com/ssaru/convert2Yolo.git

   $ cd convert2Yolo

   $ python3 example.py # sample
   --datasets COCO
   --img_path [PATH]/input/data
   --label [PATH]/input/train.json 
   --convert_output_path [PATH]/input/data 
   --img_type ".jpg" 
   --manipast_path [PATH]/input/YOLO
   --cls_list_file [PATH]/input/YOLO/trash.names
   ```

   trash.names 에는 class names 이 포함되어있다.

3. Dataset
- kfold/kfold0.yaml (example) 

   ```yaml
   train: [PATH]/input/YOLO/train0/manifest.txt 
   val: [PATH]/input/YOLO/valid0/manifest.txt
   test: [PATH]/input/YOLO/test/manifest.txt

   # number of classes
   nc: 11

   # class names
   names: [ "UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing" ]
   ```
   

## Train

- Train 
   ```shell
   $ python3 train.py # 1 fold data
   --weights yolov5x6.pt
   --data kfold/kfold0.yaml
   --hyp config/hyp_final.yaml
   --epochs 60
   --batch 64
   --img 512
   --cache
   --project kfold
   --save_perioud 10
   --image-weight
   ```

- Test & submission
   ```shell
   $ python3 test_submit.py 
   --weights kfold/exp1/weights/best.pt
   --data kfold/kfold0.yaml
   --batch 64
   --img 512
   --project kfold
   --exist-ok
   ```

## WBF ensemble
- Check the [Notebook](p3-ims-obd-code-chaser/obj-det/yolov5/det_utils.ipynb)