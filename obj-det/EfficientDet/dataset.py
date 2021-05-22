from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
import cv2
import pandas as pd
import numpy as np
import os
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
IMAGENET_DEFAULT_MEAN = [x * 255 for x in (0.485, 0.456, 0.406)]
IMAGENET_DEFAULT_STD = [x * 255 for x in (0.229, 0.224, 0.225)]

class CustomDataset(Dataset):
    '''
      data_dir: data가 존재하는 폴더 경로
      transforms: data transform (resize, crop, Totensor, etc,,,)
    '''
    def __init__(self, annotation, data_dir, transforms):
        super().__init__()
        self.data_dir = data_dir
        # coco annotation 불러오기 (coco API)
        self.coco = COCO(annotation)
        self.predictions = {
            "images": self.coco.dataset["images"].copy(),
            "categories": self.coco.dataset["categories"].copy(),
            "annotations": None
        }
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds(imgIds=index)
        image, boxes, labels = self.load_image_and_boxes(index)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        
        # Multi Scale
        target['img_size'] = torch.tensor([(512, 512)])
        target['img_scale'] = torch.tensor([1.])
        
        if self.transforms:
            # Transform 적용 후 Box가 없다면 있을 때 까지 반복 (default = 10)
            for i in range(10):
                sample = {
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': target['labels']
                }
                sample = self.transforms(**sample)
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                    target['labels'] = torch.tensor(sample['labels'])
                    break
        
        return image, target, image_id
    
    def __len__(self) -> int:
        return len(self.coco.getImgIds())
    
    def load_image_and_boxes(self, index):
        image_id = self.coco.getImgIds(imgIds=index)
        image_info = self.coco.loadImgs(image_id)[0]
        
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD

        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        boxes = np.array([x['bbox'] for x in anns])
        
        labels = np.array([x['category_id'] for x in anns])
        #labels = torch.as_tensor(labels, dtype=torch.int64)

        # boxex (x_min, y_min, x_max, y_max)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return image, boxes, labels

def collate_fn(batch):
    return tuple(zip(*batch))

def get_test_transform():
    return A.Compose([
        A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels'])
    )