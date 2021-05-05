import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import *
from tqdm import tqdm
from model import *

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


@torch.no_grad()
def inference(dataset_path, model_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # model = smp.FPN(
    #     encoder_name = 'timm-efficientnet-b2',
    #     encoder_weights = 'noisy-student',
    #     classes = 12)
    
    model = UNet_3Plus_DeepSup_CGM(n_classes=12)

    model_path = os.path.join(model_dir, 'best5.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)

    model.eval()

    test_transform = A.Compose([
                            ToTensorV2()
                            ])

    test_path = dataset_path + '/test.json'
    dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers = 4,
        pin_memory=use_cuda,
        drop_last=False,
        collate_fn = collate_fn
    )

    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)

    with torch.no_grad():
       for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).float().to(device))
            outs = outs[0] + outs[1] + outs[2] + outs[3] + outs[4]
            oms = torch.argmax(outs, dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]

    # sample_submisson.csv 열기
    submission = pd.read_csv('/opt/ml/code/submission/sample_submission.csv', index_col=None)


    # PredictionString 대입
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    
    submission.to_csv(model_dir + "/output.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for validing (default: 1000)')
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir

    inference(data_dir, model_dir,  args)
