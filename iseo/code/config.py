from dataclasses import dataclass
import torch


@dataclass
class Config:
    DATASET_PATH = "/opt/ml/input/data"
    TRAIN_DIR = f"{DATASET_PATH}/train.json"
    VAL_DIR = f"{DATASET_PATH}/val.json"
    TEST_DIR = f"{DATASET_PATH}/test.json"
    SAVED_DIR = "/opt/ml/code/saved"
    SAVED_FILENAME = "result_dlv3_ti-effb2_ns.pt"
    SUBMISSION_DIR = "/opt/ml/code/submission"
    SUBMISSION_FILENAME = "submission_dlv3_ti-effb2_ns.csv"
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0
    NUM_EPOCHS = 20
    SEED = 21
    device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    c = Config()
    print(c.TRAIN_DIR)
