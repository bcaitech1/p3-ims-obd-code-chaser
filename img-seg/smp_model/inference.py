import time

import torch
import pandas as pd
import numpy as np
from torch.nn import Module
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A


from dataset import get_category_names, get_data_loader


def inference(
    model: Module,
    data_loader: DataLoader,
    device: str,
    submission_dir: str = "./submission",
    submission_filename: str = "submission.csv",
) -> None:
    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    print("Start prediction.")
    st = time.time()
    model.eval()
    model.to(device)

    file_name_list = []
    preds = np.empty((0, size * size), dtype=np.long)

    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(data_loader):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs, dim=1).detach().cpu().numpy()

            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed["mask"]
                temp_mask.append(mask)

            oms = np.array(temp_mask)

            oms = oms.reshape([oms.shape[0], size * size]).astype(int)
            preds = np.vstack((preds, oms))

            file_name_list.append([i["file_name"] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]

    submission = pd.read_csv(f"{submission_dir}/sample_submission.csv", index_col=None)

    for file_name, string in zip(file_names, preds):
        submission = submission.append(
            {
                "image_id": file_name,
                "PredictionString": " ".join(str(e) for e in string.tolist()),
            },
            ignore_index=True,
        )

    # submission.csv로 저장
    submission.to_csv(f"{submission_dir}/{submission_filename}", index=False)
    elapsed = time.time() - st
    print("inference done! elapsed::", elapsed)


if __name__ == "__main__":
    DATASET_PATH = "/opt/ml/input/data"
    TEST_DIR = f"{DATASET_PATH}/test.json"
    TRAIN_DIR = f"{DATASET_PATH}/train.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_loader = get_data_loader(
        "test",
        batch_size=16,
        anns_file_path=TRAIN_DIR,
        data_dir=TEST_DIR,
        dataset_path=DATASET_PATH,
    )

    model = smp.DeepLabV3Plus(encoder_name="resnet101", in_channels=3, classes=12)
    model.load_state_dict(torch.load("./saved/result.pt"))

    inference(model, test_loader, device)

