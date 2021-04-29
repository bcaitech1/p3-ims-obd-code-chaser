from dataclasses import dataclass
import torch


@dataclass
class Config:
    DATASET_PATH = "/opt/ml/input/data"
    TRAIN_DIR = f"{DATASET_PATH}/train.json"
    VAL_DIR = f"{DATASET_PATH}/val.json"
    TEST_DIR = f"{DATASET_PATH}/test.json"
    SAVED_DIR = "./saved"
    SAVED_FILENAME = "result.pt"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILENAME = "submission.csv"
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0
    NUM_EPOCHS = 20
    SEED = 21
    device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    c = Config()
    print(c.TRAIN_DIR)
