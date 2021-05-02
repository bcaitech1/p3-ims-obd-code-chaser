import os
import time
import random
from typing import Tuple

import torch
import numpy as np
from torch.nn import Module
from torch.utils.data import DataLoader
import wandb


def seed_every(random_seed: int = 21):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def train(
    num_epochs: int,
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Module,
    optimizer: Module,
    val_every: int,
    device: str,
    saved_dir: str,
    saved_filename: str = "result.pt",
) -> None:

    wandb.init()
    wandb.watch(model)

    st = time.time()
    print("Start training...")
    best_mIoU = 0.0

    step = 0
    for epoch in range(num_epochs):
        model.train()
        for images, masks, _ in train_loader:
            step += 1
            images = torch.stack(images)  # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)

            images, masks = images.to(device), masks.to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 25 == 0:
                loss_train = loss.item()
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{step}/{len(train_loader)}], Loss: {loss_train:.4f}"
                )

                wandb.log({"Epoch": epoch + 1, "Step": step, "Train Loss": loss_train})

        if (epoch + 1) % val_every == 0:
            avgr_loss, mIoU = validation(
                epoch + 1, model, val_loader, criterion, device
            )
            if mIoU > best_mIoU:
                print(f"Best performance at epoch: {epoch+1}")
                print("Save model in", saved_dir)
                best_mIoU = mIoU
                save_model(model, saved_dir, saved_filename)

            wandb.log({"Epoch": epoch + 1, "Val Loss": avgr_loss, "Val mIoU": mIoU})

    elapsed = time.time() - st
    print(f"training Done! elapsed:: {elapsed}s")


def validation(
    epoch: int,
    model: Module,
    data_loader: DataLoader,
    criterion: Module,
    device: str,
    n_class: int = 12,
):
    print(f"Start validation #{epoch}")
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):

            images = torch.stack(images)
            masks = torch.stack(masks).long()

            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()

            hist = add_hist(
                hist, masks.detach().cpu().numpy(), outputs, n_class=n_class
            )

            mIoU = label_accuracy_score(hist)[2]

        avgr_loss = total_loss / cnt
        print(f"Validation #{epoch} Average Loss: {avgr_loss:.4f}, mIoU: {mIoU:.4f}")

    return avgr_loss, mIoU


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def add_hist(hist, label_trues, label_preds, n_class):
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    return hist


def label_accuracy_score(hist) -> Tuple[float, float, float, float]:
    """Returns accuracy score evaluation result.
        - overall accuracy
        - mean accuracy
        - mean IU
        - fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide="ignore", invalid="ignore"):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return acc, acc_cls, mean_iu, fwavacc


def save_model(model: Module, saved_dir: str, filename: str):
    output_path = os.path.join(saved_dir, filename)
    torch.save(model.state_dict(), output_path)
