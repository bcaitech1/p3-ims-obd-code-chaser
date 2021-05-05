# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
import numpy as np
import random
import time
import torch
import torch.nn as nn
import json

def _fast_hist(label_true, label_pred, n_class):
    """
    label_true : (H*W,)
    label_pred : (H*W,)
    """
    mask = (label_true >= 0) & (label_true < n_class) #(H*W,)
    hist = np.bincount( #  0부터 가장 큰 값까지 각각의 발생 빈도수를 체크
        n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist
    #https://gaussian37.github.io/vision-segmentation-miou/

# def label_accuracy_score(label_trues, label_preds, n_class):
#     """Returns accuracy score evaluation result.
#       - overall accuracy
#       - mean accuracy
#       - mean IU
#       - fwavacc
#       label_trues : (batch, H, W)
#       label_preds : (batch, H, W)
#     """
#     hist = np.zeros((n_class, n_class))  #(n_class,n_class)
#     for lt, lp in zip(label_trues, label_preds):
#         hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
#     acc = np.diag(hist).sum() / hist.sum()
#     with np.errstate(divide='ignore', invalid='ignore'):
#         acc_cls = np.diag(hist) / hist.sum(axis=1)
#     acc_cls = np.nanmean(acc_cls)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         iu = np.diag(hist) / (
#             hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
#         )
#     mean_iu = np.nanmean(iu)
#     freq = hist.sum(axis=1) / hist.sum()
#     fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
#     return acc, acc_cls, mean_iu, fwavacc

def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - mean IU
      label_trues : (batch, H, W)
      label_preds : (batch, H, W)
    """
    mean_iu = []
    for lt, lp in zip(label_trues, label_preds):
        hist = _fast_hist(lt.flatten(), lp.flatten(), n_class)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
            )
        mean_iu.append(np.nanmean(iu))

    mean_iu = np.nanmean(mean_iu)    
    return mean_iu


def set_seed(random_seed=21):
    # seed 고정
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
 
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mIoU_max = 0
        self.delta = delta
        self.path = path

    def __call__(self, mIoU, model):

        score = mIoU

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(mIoU, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(mIoU, model)
            self.counter = 0

    def save_checkpoint(self, mIoU, model):
        if self.verbose:
            print(f'Validation mIou increased ({self.mIoU_max:.6f} --> {mIoU:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.mIoU_max = mIoU


class ModelSave:
    def __init__(self, path):
 
        self.mIoU_max = -987654321
        self.loss_min = 987654321
        self.path = path
        

    def __call__(self, mIoU, loss, model):

        if self.mIoU_max == -987654321 and self.loss_min == 987654321:
            self.save_checkpoint(mIoU, loss, model, 'all')
            self.mIoU_max = mIoU
            self.loss_min = loss

        elif mIoU > self.mIoU_max  and loss < self.loss_min :
            self.save_checkpoint(mIoU, loss, model, 'all')
            self.mIoU_max = mIoU
            self.loss_min = loss

        elif mIoU > self.mIoU_max:
            self.save_checkpoint(mIoU, loss, model, 'mIoU')
            self.mIoU_max = mIoU

        elif loss < self.loss_min: 
            self.save_checkpoint(mIoU, loss, model, 'loss')
            self.loss_min = loss


    def save_checkpoint(self, mIoU, loss, model, save_type):
        if save_type == 'all':
            print(f'all save.    mIoU: ({self.mIoU_max:.6f} --> {mIoU:.6f})  loss: ({self.loss_min:.6f} --> {loss:.6f})  Saving model ...')
            torch.save(model.state_dict(), self.path + '/best_all.pth')

        elif save_type == 'mIoU':
            print(f'mIoU save.   mIoU: ({self.mIoU_max:.6f} --> {mIoU:.6f})   Saving model ...    min_loss: ({self.loss_min:.6f})')
            torch.save(model.state_dict(), self.path + '/best_mIoU.pth')

    
        elif save_type == 'loss':
            print(f'loss save.   loss: ({self.loss_min:.6f} --> {loss:.6f})   Saving model ...    max_mIoU: ({self.mIoU_max:.6f})')
            torch.save(model.state_dict(), self.path + '/min_loss.pth')
