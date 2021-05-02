import os
import json
import sys
from typing import List

import cv2
import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import numpy as np


def get_category_names(dataset_path: str) -> List[str]:
    with open(dataset_path, "r") as f:
        dataset = json.loads(f.read())

    return ["Background"] + [c["name"] for c in dataset["categories"]]


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]["id"] == classID:
            return cats[i]["name"]
    return "None"


class CustomDataset(Dataset):
    """COCO format"""

    def __init__(
        self,
        data_dir: str,
        category_names: List[str],
        dataset_path: str,
        mode: str = "train",
        transform=None,
    ):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        self.dataset_path = dataset_path
        self.category_names = category_names

    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        # cv2를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.dataset_path, image_infos["file_name"]))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0

        if self.mode in ("train", "val"):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos["id"])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks: size가 (height x width) 인 2D
            # 각각의 pixel 값에는 "category id + 1" 할당
            # Background  = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # Unknown = 1, Generatl trash = 2, ..., Cigarette = 11
            for i in range(len(anns)):
                className = get_classname(anns[i]["category_id"], cats)
                pixel_value = self.category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i]) * pixel_value, masks)
            masks = masks.astype(np.float32)

            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]

            return images, masks, image_infos

        if self.mode == "test":
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]

            return images, image_infos

    def __len__(self) -> int:
        return len(self.coco.getImgIds())


def _collate_fn(batch):
    return tuple(zip(*batch))


def _get_category_names(dataset_path: str) -> List[str]:
    with open(dataset_path, "r") as f:
        dataset = json.loads(f.read())

    return ["Background"] + [c["name"] for c in dataset["categories"]]


def get_data_loader(
    mode: str,
    batch_size: int,
    anns_file_path: str,
    data_dir: str,
    dataset_path: str,
    transform=None,
) -> DataLoader:

    category_names = _get_category_names(anns_file_path)

    if transform is None:
        transform = A.Compose([ToTensorV2()])

    dataset = CustomDataset(
        mode=mode,
        data_dir=data_dir,
        category_names=category_names,
        dataset_path=dataset_path,
        transform=transform,
    )

    return DataLoader(
        dataset=dataset,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=_collate_fn,
        drop_last=mode in ("train", "val"),
    )


if __name__ == "__main__":
    anns_file_path = "/opt/ml/input/data/train.json"
    dataset_path = "/opt/ml/input/data"
    train_dir = f"{dataset_path}/train.json"
    category_names = get_category_names(anns_file_path)

    train_loader = get_data_loader(
        "train",
        batch_size=4,
        anns_file_path=anns_file_path,
        data_dir=train_dir,
        dataset_path=dataset_path,
    )

    for imgs, masks, image_infos in train_loader:
        image_infos = image_infos[0]
        temp_images = imgs
        temp_masks = masks

        break

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))

    print("image shape:", list(temp_images[0].shape))
    print("mask shape:", list(temp_masks[0].shape))
    print(
        "Unique values, category of transformed mask : \n ",
        [{int(i), category_names[int(i)]} for i in list(np.unique(temp_masks[0]))],
    )

    ax1.imshow(temp_images[0])
    # ax1.imshow(temp_images[0].permute([1, 2, 0]))
    ax1.grid(False)
    ax1.set_title(f"input image: {image_infos['file_name']}", fontsize=15)

    ax2.imshow(temp_masks[0])
    ax2.grid(False)
    ax2.set_title(f"masks :{image_infos['file_name']}", fontsize=15)

    plt.savefig("ret.png")

