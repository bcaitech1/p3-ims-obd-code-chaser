import gluoncv.utils as gcv
import torch
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from datetime import datetime
import time
import random
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import warnings
import argparse
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
from dataset import *
from utils import *

def train_fn(model, train_data_loader, val_data_loader, optimizer, scheduler, device, args):
    model.train()
    train_loss = AverageMeter()
    train_cls_loss = AverageMeter()
    train_box_loss = AverageMeter()
    save_path = './checkpoint.pth'
    early_stop = EarlyStopping(path=save_path)
    for epoch in range(args.epochs):
        train_loss.reset()
        train_cls_loss.reset()
        train_box_loss.reset()
        model.train()
        for images, targets, image_ids in tqdm(train_data_loader):
            # gpu 계산을 위해 image.to(device)
            images = torch.stack(images)
            images = images.to(device).float()
            current_batch_size = images.shape[0]

            targets2 = {}
            targets2['bbox'] = [target['boxes'].to(device).float() for target in targets] # variable number of instances, so the entire structure can be forced to tensor
            targets2['cls'] = [target['labels'].to(device).float() for target in targets]
            targets2['image_id'] = torch.tensor([target['image_id'] for target in targets]).to(device).float()
            targets2['img_scale'] = torch.tensor([target['img_scale'] for target in targets]).to(device).float()
            targets2['img_size'] = torch.tensor([(512, 512) for target in targets]).to(device).float()

            # calculate loss
            losses, cls_loss, box_loss = model(images, targets2).values()

            train_loss.update(losses.detach().item(), current_batch_size)
            train_cls_loss.update(cls_loss.detach().item(), current_batch_size)
            train_box_loss.update(box_loss.detach().item(), current_batch_size)

            # backward
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        scheduler.step()
        
        # Evaluation
        val_loss, val_mAP = vali_fn(val_data_loader,model,device)
        
        print(f"\nEpoch #{epoch+1} train loss: [{train_loss.avg:.4f}] train cls loss : [{train_cls_loss.avg:.4f}] train box loss : [{train_box_loss.avg:.4f}] validation mAP : [{val_mAP:.4f}] ")
        print(f"learning rate :{scheduler.get_lr()} \n")
        #wandb.log({"Train Loss":train_loss.avg, "Validation mAP@50":val_mAP, "LR":scheduler.get_lr()})
        early_stop(val_mAP,model)
        if early_stop.early_stop:
            print('Stop Training.....')
            break

def vali_fn(val_data_loader, model, device):
    model.eval()
    vali_loss = AverageMeter()
    vali_mAP = AverageMeter()
    # Custom
    metric = gcv.metrics.VOCMApMetric(iou_thresh=0.5)
    with torch.no_grad():
        for images, targets, image_ids in tqdm(val_data_loader):
            # gpu 계산을 위해 image.to(device)
            images = torch.stack(images)
            images = images.to(device).float()

            current_batch_size = images.shape[0]

            targets2 = {}
            targets2['bbox'] = [target['boxes'].to(device).float() for target in targets] # variable number of instances, so the entire structure can be forced to tensor
            targets2['cls'] = [target['labels'].to(device).float() for target in targets]
            targets2['image_id'] = torch.tensor([target['image_id'] for target in targets]).to(device).float()
            targets2['img_scale'] = torch.tensor([target['img_scale'] for target in targets]).to(device).float()
            targets2['img_size'] = torch.tensor([(512, 512) for target in targets]).to(device).float()

            outputs = model(images, targets2)

            loss = outputs['loss']
            det = outputs['detections']

            # Calc Metric
            for i in range(0, len(det)):
                pred_scores=det[i, :, 4].cpu().unsqueeze_(0).numpy()
                condition=(pred_scores > 0.05)[0]
                gt_boxes=targets2['bbox'][i].cpu().unsqueeze_(0).numpy()[...,[1,0,3,2]] #move to PASCAL VOC from yxyx format
                gt_labels=targets2['cls'][i].cpu().unsqueeze_(0).numpy()

                pred_bboxes=det[i, :, 0:4].cpu().unsqueeze_(0).numpy()[:, condition, :]
                pred_labels=det[i, :, 5].cpu().unsqueeze_(0).numpy()[:, condition]
                pred_scores=pred_scores[:, condition]
                metric.update(
                  pred_bboxes=pred_bboxes,
                  pred_labels=pred_labels,
                  pred_scores=pred_scores,
                  gt_bboxes=gt_boxes,
                  gt_labels=gt_labels)

            vali_mAP.update(metric.get()[1], current_batch_size)
            vali_loss.update(loss.detach().item(), current_batch_size)
    
    # validation loss
    return vali_loss.avg, vali_mAP.avg

def get_train_transform():
    return A.Compose([
        A.RandomResizedCrop(384, 384, p=0.3),
        A.Resize(512, 512),
        A.ShiftScaleRotate(rotate_limit=30,p=0.4),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels'])
    )

def get_valid_transform():
    return A.Compose([
        A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels'])
    )

def get_model(check_dir = None):
    # get weight
    # url : https://github.com/rwightman/efficientdet-pytorch
    config = get_efficientdet_config('tf_efficientdet_d6')
    config.image_size = (512, 512)
    config.norm_kwargs=dict(eps=.001, momentum=.01)
    net = EfficientDet(config, pretrained_backbone=True)
    if check_dir != None:
        checkpoint = torch.load(check_dir)
        net.load_state_dict(checkpoint)

    net.reset_head(num_classes=11)
    net.class_net = HeadNet(config, num_outputs=11)
    return DetBenchTrain(net, config)

if __name__ == '__main__':
    warnings.filterwarnings(action='ignore') 

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'device is :{device}')
    
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--train_batch_size', type=int, default=6,
                        help='input batch size for training (default: 6)')
    parser.add_argument('--valid_batch_size', type=int, default=6,
                        help='input batch size for validing (default: 6)')
    parser.add_argument('--model', type=str, default='BaseModel',
                        help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='adamW',
                        help='optimizer type (default: adamW)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate (default: 3e-4)')

    args = parser.parse_args()
    seed_everything(args.seed)
    
    train_annotation = '../input/data/train_all.json'
    val_annotation = '../input/data/val.json'
    data_dir = '../input/data'
    train_dataset = CustomDataset(train_annotation, data_dir, get_train_transform())
    val_dataset = CustomDataset(val_annotation, data_dir, get_valid_transform())

    train_data_loader=DataLoader(train_dataset,batch_size=args.train_batch_size,shuffle=True,collate_fn=collate_fn,num_workers=4)
    val_data_loader=DataLoader(val_dataset,batch_size=args.valid_batch_size,shuffle=False,collate_fn=collate_fn,num_workers=4)
    
    # Model
    model = get_model()
    model.to(device)
    # setting
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-4)
    
    train_fn(model, train_data_loader, val_data_loader, optimizer, scheduler, device, args)