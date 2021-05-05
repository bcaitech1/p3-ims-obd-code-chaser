import argparse
import json
import os
import random
import re
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import *
from adamp import AdamP
from loss import FocalLoss, JaccardLoss, DiceLoss, loss_sum, SoftCrossEntropyLoss
from tqdm import tqdm
from utils import *
from model import *

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import time


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(dataset_path, model_dir, args):
    save_dir = increment_path(os.path.join(model_dir, args.name))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- augmentation
    train_transform = A.Compose([
        # A.ShiftScaleRotate(),
        # A.RandomBrightnessContrast(),
        # A.HorizontalFlip(),
        # A.VerticalFlip(),
        # A.RandomResizedCrop(512, 512, (0.75, 1.0), p=0.5),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        ToTensorV2()
    ])
   
    train_path = dataset_path + '/train.json'
    val_path = dataset_path + '/val.json'

    # -- data_loader
    train_set = CustomDataLoader(data_dir=train_path, mode='train', transform=train_transform)
    val_set = CustomDataLoader(data_dir=val_path, mode='val', transform=val_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
        collate_fn = collate_fn
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
        collate_fn = collate_fn
    )

    model = UNet_3Plus_DeepSup_CGM(n_classes=12)
    model = model.to(device)

    # -- loss & metric
    criterion = SoftCrossEntropyLoss(smooth_factor= 0.1)
    criterion2 = nn.BCELoss()

    optimizer = AdamP(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    early_stopping = [EarlyStopping(patience=7, verbose=True, path = save_dir+f'/best{i}.pth') for i in range(7)]
#    model_save = ModelSave(path = save_dir)

    start = time.time()
    for epoch in range(args.epochs):
        # train loop
        model.train()

        for idx, (images, masks, label) in enumerate(train_loader):
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)
            label = torch.stack(label)

            # gpu 연산을 위해 device 할당
            images, masks, label = images.to(device), masks.to(device), label.to(device)

            optimizer.zero_grad()
            outputs = model(images)  

            loss = 0
            for output in outputs[:-1]:
                loss += criterion(output, masks)
            
            loss += criterion2(outputs[-1], label)


            loss.backward()
            optimizer.step()

            if (idx + 1) % 20 == 0:
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || loss {loss.item():.4f} || lr {current_lr}" 
                )
                logger.add_scalar("Train/loss", loss, epoch * len(train_loader) + idx)

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            total_loss = 0
            cnt = 0
            mIoU_list = [[], [], [], [], [], [], []]
            for (images, masks, label) in val_loader:
                images = torch.stack(images)       # (batch, channel, height, width)
                masks = torch.stack(masks).long()  # (batch, height, width)
                label = torch.stack(label)

                images, masks, label = images.to(device), masks.to(device), label.to(device)

                outputs = model(images) # (batch, n_classes, height, width)

                loss = 0
                for output in outputs[:-1]:
                    loss += criterion(output, masks)

                loss += criterion2(outputs[-1], label)

                total_loss += loss.item()
                cnt += 1

                for idx, output in enumerate(outputs[:-1]):
                    output = torch.argmax(output, dim=1).detach().cpu().numpy()
                    mIoU = label_accuracy_score(masks.detach().cpu().numpy(), output, n_class=12)
                    mIoU_list[idx].append(mIoU)
                

                outputs1 = outputs[0] + outputs[1] + outputs[2] + outputs[3] + outputs[4]
                outputs1 = torch.argmax(outputs1.squeeze(), dim=1).detach().cpu().numpy()
                mIoU = label_accuracy_score(masks.detach().cpu().numpy(), outputs1, n_class=12)
                mIoU_list[5].append(mIoU)

                outputs2 = outputs[0] + outputs[1] + outputs[2] + outputs[3]
                outputs2 = torch.argmax(outputs2.squeeze(), dim=1).detach().cpu().numpy()
                mIoU = label_accuracy_score(masks.detach().cpu().numpy(), outputs2, n_class=12)
                mIoU_list[6].append(mIoU)

            avrg_loss = total_loss / cnt
            mIoU = [np.mean(mIoU_list[iou_idx]) for iou_idx in range(7)]

            print(f'Validation #{epoch}  Average Loss: {avrg_loss:.4f}')
            for iou_idx in range(7):
                print(f"depth {iou_idx} : {mIoU[iou_idx]:.4f}", end=" || ")
            print()
            torch.save(model.state_dict(), f"{save_dir}/last.pth")
            
            logger.add_scalar("Val/loss", avrg_loss, epoch)
            for iou_idx in range(7):
                print("###################################")
                print(f"depth : {iou_idx}")
                logger.add_scalar(f"Val/mIoU{iou_idx}", mIoU[iou_idx], epoch)
                early_stopping[iou_idx](mIoU[iou_idx], model)
            print("###################################")

            #early_stopping(mIoU, model)
            #model_save(mIoU = mIoU, loss = avrg_loss, model = model)
            # if early_stopping.early_stop:
            #     print('Early stopping')
            #     break
            print()

    end = time.time()
    print(f'time : {end-start}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=21, help='random seed (default: 21)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate (default: 3e-4)')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    set_seed(args.seed)

    torch.cuda.empty_cache()

    train(data_dir, model_dir,args)
        
#tensorboard --logdir=./ --host=0.0.0.0 --port=6006