import json
from typing import List

import wandb
import torch
from torch import nn
import segmentation_models_pytorch as smp

import albumentations as A
from albumentations.pytorch import ToTensorV2
from transform import *
from dataset import get_data_loader
from train import train, seed_every
from inference import inference
from loss import FocalLoss
from adamp import AdamP
from pytorch_toolbelt import losses as L
import argparse

model_dict = {
    "Unet": smp.Unet, 
    "UnetPlusPlus": smp.UnetPlusPlus, 
    "FPN": smp.FPN, 
    "PSPNet": smp.PSPNet, 
    "DeepLabV3": smp.DeepLabV3, 
    "DeepLabV3Plus": smp.DeepLabV3Plus 
}
encoder_dict = {
    "resnet18": resnet18, # imagenet / ssl / swsl # 11M
    "resnet34": resnet34, # imagenet # 21M
    "resnet50": resnet50, # imagenet / ssl / swsl # 23M
    "resnext50_32x4d": resnext50_32x4d, # ssl / swsl # 86M
    "efficientnetb0": timm-efficientnet-b0, # imagenet / advprop / noisy-student # 4M
    "efficientnetb2": timm-efficientnet-b2, # imagenet / advprop / noisy-student # 7M
    "efficientnetb5": timm-efficientnet-b5, # imagenet / advprop / noisy-student # 28M
    "efficientnetb7": timm-efficientnet-b7, # imagenet / advprop / noisy-student # 63M
    "efficientnetl2": timm-efficientnet-l2, # noisy-student # 474M
}
pretrained_dict = {
    "imagenet": imagenet, 
    "ssl": ssl, 
    "swsl": swsl, 
    "advprop": advprop, 
    "noisy-student": noisy-student, 

}

def build_model(model_name, encoder_name,pretrained_name, num_classes=12, name="model"):
    assert model_name in model_dict.keys(), f"Please, check pretrained model list {list(model_dict.keys())}"
    assert encoder_name.lower() in encoder_dict.keys(), f"Please, check pretrained encoder list {list(encoder_dict.keys())}"
    assert encoder_name.lower() in encoder_dict.keys(), f"Please, check pretrained encoder list {list(encoder_dict.keys())}"

    activation = "softmax" if num_classes > 1 else "sigmoid"

    
    model = model_dict[model_name](
        encoder_name=encoder_dict[encoder_name],
        encoder_weights=pretrained_dict[pretrained_name],
        in_channels=3,
        classes=num_classes,
        activation=activation,
        )
            
    return model
    
# define your params

def get_config() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/opt/ml/input/data")
    parser.add_argument("--save", type=str, default="/opt/ml/code/saved")
    parser.add_argument("--savedname", type=str, default="best.pt")
    parser.add_argument("--submit", type=str, default="/opt/ml/code/submission")
    parser.add_argument("--submitname", type=str, default="submission.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.002)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=21)

    parser.add_argument("--model", type=str, default="deeplabv3")
    parser.add_argument("--encoder", type=str, default="efficientnetb0")
    parser.add_argument("--pretrained", type=str, default="imagenet")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return Config(
        DATASET_PATH = args.data,
        SAVED_DIR = args.save,
        SAVED_FILENAME = args.savedname,
        SUBMISSION_DIR = args.submit,
        SUBMISSION_FILENAME = args.submitname,
        TRAIN_DIR = args.data / "train.json",
        VAL_DIR = args.data / "val.json",
        TEST_DIR = args.data / "test.json",
        BATCH_SIZE = args.batch,
        LEARNING_RATE = args.lr,
        WEIGHT_DECAY = args.wd,
        NUM_EPOCHS = args.epochs,
        SEED = args.seed,
        device = device,
        MODEL = args.model,
        ENCODER = args.encoder,
        PRETRAINED = args.pretrained,

    )


def main(c) -> None:
    seed_every(c.SEED)

    model = build_model(
            model_name=c.MODEL,
            encoder_name=c.ENCODER,
            pretrained_name=c.PRETRAiNED,
            num_classes=12,
            )
    # model.load_state_dict(torch.load(f"{c.SAVED_DIR}/{c.SAVED_FILENAME}"))
    model = model.to(c.device)

    # criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(gamma=0.5)
    criterion = L.JaccardLoss('multiclass')

    # optimizer = MADGRAD(

    optimizer = AdamP(params = model.parameters(),
                        lr = c.LEARNING_RATE,
                        weight_decay=c.WEIGHT_DECAY,
                        betas=(0.9, 0.999))

    train_transform = A.Compose([ToTensorV2()])
    val_transform = A.Compose([ToTensorV2()])

    train_loader = get_data_loader(
        "train",
        batch_size=c.BATCH_SIZE,
        anns_file_path=c.TRAIN_DIR,
        data_dir=c.TRAIN_DIR,
        dataset_path=c.DATASET_PATH,
        transform=train_augmentation(),
    )

    val_loader = get_data_loader(
        "val",
        batch_size=c.BATCH_SIZE,
        anns_file_path=c.TRAIN_DIR,
        data_dir=c.VAL_DIR,
        dataset_path=c.DATASET_PATH,
        transform=train_augmentation(),
    )

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
    print(f"PyTorch version:[{torch.__version__}].")
    c = get_config()
    print(f"This code use [{cfg.DEVICE}]")
    main(c)
