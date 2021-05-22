import albumentations as A
from albumentations.pytorch import ToTensorV2
from gridmask import *

def train_augmentation():
    train_transform = [

        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),              
        A.Rotate(p=0.5, limit=(-30, 30), interpolation=0, border_mode=4), 
        
        A.OneOf(
            [
                A.RandomResizedCrop(512, 512, (0.75, 1.0), p=1),
                GridMask(num_grid=3, mode=0, p=1),
                GridMask(num_grid=4, mode=0, p=1),
                GridMask(num_grid=5, mode=0, p=1)
            ],
            p=0.5,
        ),
                
        ToTensorV2()
    ]
    return A.Compose(train_transform)

def val_augmentation():
    transform = A.Compose([
                            ToTensorV2()
                            ])
    return transform

def test_augmentation():
    transform = A.Compose([
                            ToTensorV2()
                            ])
    return transform