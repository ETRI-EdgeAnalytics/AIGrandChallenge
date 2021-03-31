import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import math

'''
Label Smoothing described in "Rethinking the Inception Architecture for Computer Vision"
Ref: https://github.com/PistonY/torch-toolbox/blob/master/torchtoolbox/nn/loss.py
     https://github.com/whr94621/NJUNMT-pytorch/blob/master/src/modules/criterions.py
'''
class LabelSmoothingLoss(nn.Module):
    def __init__(self, device, classes, smoothing=0.0, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls= classes
        self.dim = dim
        self.device = device

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred).to(self.device)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


'''
Smooth CrossEntropyLoss
Ref: https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/smooth_ce.py
'''
class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, device, label_smoothing=0.0, size_average=True):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.size_average = size_average
        self.device = device

    def cross_entropy(self, pred, target, size_average=True):
        """ Cross entropy that accepts soft targets
        Args:
             pred: predictions for neural network
             targets: targets, can be soft
             size_average: if false, sum is returned instead of mean
        Examples::
            input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
            input = torch.autograd.Variable(out, requires_grad=True)
            target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
            target = torch.autograd.Variable(y1)
            loss = cross_entropy(input, target)
            loss.backward()
        """
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        if size_average:
            return torch.mean(torch.sum(-target * logsoftmax(pred), dim=1))
        else:
            return torch.sum(torch.sum(-target * logsoftmax(pred), dim=1))

    def forward(self, pred, target):
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=pred.size(-1))
            target = target.float().to(self.device)
        if self.label_smoothing > 0.0:
            s_by_c = self.label_smoothing / len(pred[0])
            smooth = torch.zeros_like(target)
            smooth = smooth + s_by_c
            target = target * (1. - s_by_c) + smooth

        return self.cross_entropy(pred, target, self.size_average)

'''
Cutmix CrossEntropyLoss
Ref: https://github.com/ildoonet/cutmix/blob/master/cutmix/utils.py
'''
class CutMixCrossEntropyLoss(nn.Module):
    def __init__(self, device, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.device = device

    def cross_entropy(self, pred, target, size_average=True):
        """ Cross entropy that accepts soft targets
        Args:
             pred: predictions for neural network
             targets: targets, can be soft
             size_average: if false, sum is returned instead of mean
        Examples::
            input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
            input = torch.autograd.Variable(out, requires_grad=True)
            target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
            target = torch.autograd.Variable(y1)
            loss = cross_entropy(input, target)
            loss.backward()
        """
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        if size_average:
            return torch.mean(torch.sum(-target * logsoftmax(pred), dim=1))
        else:
            return torch.sum(torch.sum(-target * logsoftmax(pred), dim=1))

    def forward(self, pred, target):
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=pred.size(-1))
            target = target.float().to(self.device)
        return self.cross_entropy(pred,target,self.size_average)

# cross entropy loss with label smoothing
# https://github.com/etetteh/sota-data-augmentation-and-optimizers
# https://github.com/eladhoffer/utils.pytorch/blob/master/cross_entropy.py
class CrossEntropyLossWithLabelSmoothing(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets, and optional label smoothing"""

    def __init__(self, device, weight=None, ignore_index=-100, reduction='mean', smooth_eps=None, smooth_dist=None, from_logits=True):
        super(CrossEntropyLossWithLabelSmoothing, self).__init__(weight=weight,
                                               ignore_index=ignore_index, reduction=reduction)
        self.smooth_eps = smooth_eps
        self.smooth_dist = smooth_dist
        self.from_logits = from_logits

        self.device = device

    def onehot(self,indexes, N=None, ignore_index=None):
        """
        Creates a one-representation of indexes with N possible entries
        if N is not specified, it will suit the maximum index appearing.
        indexes is a long-tensor of indexes
        ignore_index will be zero in onehot representation
        """
        if N is None:
            N = indexes.max() + 1
        sz = list(indexes.size())
        output = indexes.new().byte().resize_(*sz, N).zero_()
        output.scatter_(-1, indexes.unsqueeze(-1), 1)
        if ignore_index is not None and ignore_index >= 0:
            output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
        return output

    def _is_long(self,x):
        if hasattr(x, 'data'):
            x = x.data
        return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)


    def cross_entropy(self,inputs, target, weight=None, ignore_index=-100, reduction='mean',
                  smooth_eps=None, smooth_dist=None, from_logits=True):
        """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""
        smooth_eps = smooth_eps or 0

        # ordinary log-liklihood - use cross_entropy from nn
        if self._is_long(target) and smooth_eps == 0:
            if from_logits:
                return F.cross_entropy(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)
            else:
                return F.nll_loss(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)

        if from_logits:
            # log-softmax of inputs
            lsm = F.log_softmax(inputs, dim=-1)
        else:
            lsm = inputs

        masked_indices = None
        num_classes = inputs.size(-1)

        if self._is_long(target) and ignore_index >= 0:
            masked_indices = target.eq(ignore_index)

        if smooth_eps > 0 and smooth_dist is not None:
            if _is_long(target):
                target = self.onehot(target, num_classes).type_as(inputs)
            if smooth_dist.dim() < target.dim():
                smooth_dist = smooth_dist.unsqueeze(0)
            target.lerp_(smooth_dist, smooth_eps)

        if weight is not None:
            lsm = lsm * weight.unsqueeze(0)

        if self._is_long(target):
            eps_sum = smooth_eps / num_classes
            eps_nll = 1. - eps_sum - smooth_eps
            likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
            loss = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))
        else:
            loss = -(target * lsm).sum(-1)

        if masked_indices is not None:
            loss.masked_fill_(masked_indices, 0)

        if reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'mean':
            if masked_indices is None:
                loss = loss.mean()
            else:
                loss = loss.sum() / float(loss.size(0) - masked_indices.sum())

        return loss

    def forward(self, input, target, smooth_dist=None):
        if smooth_dist is None:
            smooth_dist = self.smooth_dist
        return self.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index,
                             reduction=self.reduction, smooth_eps=self.smooth_eps,
                             smooth_dist=smooth_dist, from_logits=self.from_logits)

def get_loss_fn(conf):
    loss = None  # default: CrossEntropyLoss
    if conf.get()['model']['criterion']['name'] == 'mse':
        loss = nn.MSELoss()
    elif conf.get()['model']['criterion']['name'] == 'cross_entropy':
        loss = nn.CrossEntropyLoss()
    elif conf.get()['model']['criterion']['name'] == 'label_smoothing':
        loss = LabelSmoothingLoss(conf.get()['cuda']['device'],conf.get()['model']['num_class'], conf.get()['model']['criterion']['smoothing'])
    elif conf.get()['model']['criterion']['name'] == 'smooth_cross_entropy':
        loss = SmoothCrossEntropyLoss(conf.get()['cuda']['device'],conf.get()['model']['criterion']['smoothing'])
    elif conf.get()['model']['criterion']['name'] == 'cutmix_cross_entropy':
        loss = CutMixCrossEntropyLoss(conf.get()['cuda']['device'],True)
    elif conf.get()['model']['criterion']['name'] == 'cross_entropy_with_label_smoothing':
        loss = CrossEntropyLossWithLabelSmoothing(conf.get()['cuda']['device'], smooth_eps=0.1)
    else:
        raise ValueError(conf.get()['model']['criterion']['name'])

    if conf.get()['cuda']['avail'] == True:
        loss = loss.to(torch.device(conf.get()['cuda']['device']))
    return loss

