import json
from typing import List

import wandb
import torch
from torch import nn
import segmentation_models_pytorch as smp
import torchsummary
from madgrad import MADGRAD
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import get_data_loader
from train import train, seed_every
from inference import inference
from config import Config


def main(c: Config) -> None:
    seed_every(c.SEED)

    model = smp.DeepLabV3Plus(encoder_name="resnet101", in_channels=3, classes=12).to(
        c.device
    )

    criterion = nn.CrossEntropyLoss()
    # optimizer = MADGRAD(
    #     params=model.parameters(), lr=c.LEARNING_RATE, weight_decay=c.WEIGHT_DECAY
    # )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=c.LEARNING_RATE, weight_decay=c.WEIGHT_DECAY
    )

    train_transform = A.Compose([ToTensorV2()])
    val_transform = A.Compose([ToTensorV2()])
    train_loader = get_data_loader(
        "train",
        batch_size=c.BATCH_SIZE,
        anns_file_path=c.TRAIN_DIR,
        data_dir=c.TRAIN_DIR,
        dataset_path=c.DATASET_PATH,
        transform=train_transform,
    )

    val_loader = get_data_loader(
        "val",
        batch_size=c.BATCH_SIZE,
        anns_file_path=c.TRAIN_DIR,
        data_dir=c.VAL_DIR,
        dataset_path=c.DATASET_PATH,
        transform=val_transform,
    )
    model.cuda()

    train(
        c.NUM_EPOCHS,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        1,
        c.device,
        c.SAVED_DIR,
        saved_filename=c.SAVED_FILENAME,
    )

    test_transform = A.Compose([ToTensorV2()])

    test_loader = get_data_loader(
        "test",
        c.BATCH_SIZE,
        anns_file_path=c.TRAIN_DIR,
        data_dir=c.TEST_DIR,
        dataset_path=c.DATASET_PATH,
    )

    model.load_state_dict(torch.load(f"{c.SAVED_DIR}/{c.SAVED_FILENAME}"))

    inference(model, test_loader, c.device, c.SUBMISSION_DIR, c.SUBMISSION_FILENAME)


if __name__ == "__main__":
    c = Config()
    main(c)
