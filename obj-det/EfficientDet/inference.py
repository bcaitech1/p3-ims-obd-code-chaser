import gluoncv.utils as gcv
import torch
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
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
from dataset import *

def get_saved_model(check_dir):
    config = get_efficientdet_config('tf_efficientdet_d6')
    config.image_size = (512, 512)
    config.norm_kwargs=dict(eps=.001, momentum=.01)
    net = EfficientDet(config, pretrained_backbone=False)
    net.reset_head(num_classes=11)
    net.class_net = HeadNet(config, num_outputs=11)
    
    net = DetBenchPredict(net)
    checkpoint = torch.load(check_dir)
    net.load_state_dict(checkpoint)

    return net

def test_fn(data_loader, model, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for images, targets, image_ids in tqdm(data_loader):
            # gpu 계산을 위해 image.to(device)
            images = torch.stack(images)
            images = images.to(device).float()

            det = model(images)

            for i in range(0, len(det)):
                pred_scores=det[i, :, 4].cpu().unsqueeze_(0).numpy()
                condition=(pred_scores > 0.05)[0]

                pred_bboxes=det[i, :, 0:4].cpu().unsqueeze_(0).numpy()[:, condition, :]
                pred_labels=det[i, :, 5].cpu().unsqueeze_(0).numpy()[:, condition]
                pred_scores=pred_scores[:, condition]
                
                outputs.append({'boxes':pred_bboxes,
                            'scores':pred_scores,
                            'labels':pred_labels})

    return outputs

def save_submission(outputs, args):
    score_threshold = args.threshold

    prediction_strings = []
    file_names = []
    coco = COCO(test_annotation)
    for i, output in tqdm(enumerate(outputs)):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for box, score, label in zip(output['boxes'][0], output['scores'][0], output['labels'][0]):
            if score > score_threshold:
                prediction_string += str(int(label)) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(box[1]) + ' '\
                    + str(box[2]) + ' ' + str(box[3]) + ' '

        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv('submission.csv', index=None)
    print(submission.head())

if __name__ == '__main__':
    warnings.filterwarnings(action='ignore') 

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'device is :{device}')
    
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='threshold (default: 0.3)')
    parser.add_argument('--check_dir', type=str, default='./saved/checkpoint.pth',
                        help='checkpoint directory (default: ./saved)')

    args = parser.parse_args()
    
    test_annotation = '../input/data/test.json'
    data_dir = '../input/data'
    test_dataset = CustomDataset(test_annotation, data_dir, get_test_transform())
    test_data_loader=DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    checkpoint_path = args.check_dir
    model = get_saved_model(checkpoint_path)
    model.to(device)
    
    # setting
    outputs = test_fn(test_data_loader, model, device)
    
    save_submission(outputs, args)