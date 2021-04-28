import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/TverskyLoss/binarytverskyloss.py

class FocalBinaryTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, ignore_index=None, reduction='none'):
        """Dice loss of binary class
        Args:
            alpha: controls the penalty for false positives.
            beta: penalty for false negative. Larger beta weigh recall higher
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        Shapes:
            output: A tensor of shape [N, 1,(d,) h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
        Returns:
            Loss tensor according to arg reduction
        Raise:
            Exception if unexpected reduction
        """
        super(FocalBinaryTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.smooth = 10
        self.reduction = reduction
        s = self.beta + self.alpha
        if s != 1:
            self.beta = self.beta / s
            self.alpha = self.alpha / s

    def forward(self, output, target, mask=None):
        bg_target = 1 - target

        # output = torch.sigmoid(output).view(batch_size, -1)
        output = output.contiguous().view(-1)
        target = target.contiguous().view(-1)
        bg_target = bg_target.contiguous().view(-1)

        TP = torch.sum(output * target)  # P_G
        FP = torch.sum(output * bg_target)  # P_NG
        FN = torch.sum((1 - output) * target)  # NP_G

        tversky_index = TP / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        loss = 1. - tversky_index
        # target_area = torch.sum(target_label, 1)
        # loss[target_area == 0] = 0
        # if self.reduction == 'none':
        #     loss = loss
        # elif self.reduction == 'sum':
        #     loss = torch.sum(loss)
        # else:
        #     loss = torch.mean(loss)
        return torch.pow(loss, self.gamma)






class MultiTverskyLoss(nn.Module):
    """
    Tversky Loss for segmentation adaptive with multi class segmentation
    """

    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, weights=None):
        """
        :param alpha (Tensor, float, optional): controls the penalty for false positives.
        :param beta (Tensor, float, optional): controls the penalty for false negative.
        :param gamma (Tensor, float, optional): focal coefficient
        :param weights (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`
        """
        super(MultiTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weights = weights
    
    def __name__(self):
        return "MultiTverskyLoss"

    def forward(self, inputs, targets):
        num_class = inputs.size(1)
        weight_losses = 0.0
        if self.weights is not None:
            assert len(self.weights) == num_class, 'number of classes should be equal to length of weights '
            weights = self.weights
        else:
            weights = [1.0 / num_class] * num_class
        input_slices = torch.split(inputs, [1] * num_class, dim=1)
        target_slices = torch.split(targets, [1] * num_class, dim=1)
        for idx in range(num_class):
            input_idx = input_slices[idx]
            target_idx = target_slices[idx]
            loss_func = FocalBinaryTverskyLoss(self.alpha, self.beta, self.gamma)
            loss_idx = loss_func.forward(input_idx, target_idx)
            weight_losses+=loss_idx * weights[idx]
        # loss = torch.Tensor(weight_losses)
        # loss = loss.to(inputs.device)
        # loss = torch.sum(loss)
        return weight_losses
