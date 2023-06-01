import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor
from typing import Optional, Union


class SegmentationLosses(object):
    def __init__(self,
                 weight: Optional[ndarray] = None,
                 ignoreIndex: int = 255,
                 cuda: bool = False):
        self.ignoreIndex = ignoreIndex
        self.weight = weight
        self.cuda = cuda

    def __call__(self,
                 pred: Tensor,
                 label: Tensor,
                 mode: str = 'ce',
                 alpha: Optional[float] = None,
                 gamma: float = 2,
                 classNum: int = 3,
                 reduction: str = 'mean') -> Tensor:
        if mode == 'ce':
            loss = self.crossEntropyLoss(pred, label, reduction=reduction)
        elif mode == 'focal':
            loss = self.focalLoss(pred, label, alpha=alpha, gamma=gamma, classNum=classNum, reduction=reduction)
        elif mode == 'dice':
            loss = self.diceLoss(pred, label, reduction=reduction)
        elif mode == 'gdl':
            loss = self.generalizedDiceLoss(pred, label, reduction=reduction)
        else:
            raise ValueError
        return loss

    def crossEntropyLoss(self,
                         pred: Tensor,
                         label: Tensor,
                         reduction: str = 'mean') -> Tensor:
        # logit = F.log_softmax(pred, dim=1)
        # criterion = nn.NLLLoss(self.weight, ignore_index=self.ignoreIndex)
        criterion = nn.CrossEntropyLoss(weight=self.weight,
                                        ignore_index=self.ignoreIndex,
                                        reduction=reduction)

        if self.cuda:
            criterion.cuda()

        return criterion(pred, label.long())

    def focalLoss(self,
                  pred: Tensor,
                  label: Tensor,
                  alpha: Union[float, list, ndarray] = None,
                  gamma: float = 2,
                  classNum: int = 12,
                  reduction: str = 'mean') -> Tensor:
        class FocalLoss(nn.Module):
            def __init__(self) -> None:
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.reduction = reduction
                self.classNum = classNum

                if self.alpha is None:
                    self.alpha = torch.ones(self.classNum, 1)
                elif isinstance(self.alpha, (list, ndarray)):
                    assert len(self.alpha) == self.classNum
                    self.alpha = torch.FloatTensor(self.alpha).view(self.classNum, 1)
                    self.alpha = self.alpha / self.alpha.sum()
                else:
                    raise TypeError('Not support alpha type')

            def forward(self, predict: Tensor = pred, gt: Tensor = label) -> Tensor:
                predict = F.softmax(predict, dim=1)
                epsilon = 1e-10
                pt = -F.nll_loss(predict, gt, reduction='none').view(-1) + epsilon
                logpt = pt.log()
                gt = gt.view(-1, 1)
                self.alpha = self.alpha[gt].squeeze().to(predict.device)
                loss = -1 * self.alpha * torch.pow((1 - pt), self.gamma) * logpt

                # import sys
                # temp = torch.any(torch.isnan(loss))
                # if temp:
                #     torch.set_printoptions(profile="full")
                #     print(loss)
                #     print(torch.pow((1 - pt), self.gamma))
                #     print(logpt)
                #     sys.exit(1)

                if self.reduction == 'mean':
                    loss = loss.mean()
                elif self.reduction == 'sum':
                    loss = loss.sum()
                elif self.reduction == 'none':
                    pass
                else:
                    raise ValueError("reduction must be 'mean','sum' or 'none'")

                return loss

        criterion = FocalLoss()

        if self.cuda:
            criterion.cuda()

        return criterion()

    def diceLoss(self,
                 pred: Tensor,
                 label: Tensor,
                 reduction: str = 'mean') -> Tensor:
        class DiceLoss(nn.Module):
            def __init__(self) -> None:
                super(DiceLoss, self).__init__()
                self.reduction = reduction

            def forward(self,
                        predict: Tensor = pred,
                        gt: Tensor = label,
                        epsilon: float = 1e-6) -> Tensor:
                gt = F.one_hot(gt).permute(0, 3, 1, 2)
                intersection = 2 * torch.sum(predict * gt, dim=[2, 3]) + epsilon
                union = torch.sum(predict ** 2, dim=[2, 3]) + torch.sum(gt ** 2, dim=[2, 3]) + epsilon
                loss = 1 - torch.mean(intersection / union, dim=1)
                if self.reduction == 'mean':
                    loss = torch.mean(loss, dim=0)
                elif self.reduction == 'sum':
                    loss = torch.sum(loss, dim=0)
                elif self.reduction == 'none':
                    pass
                else:
                    raise ValueError("reduction must be 'mean','sum' or 'none'")
                return loss

        criterion = DiceLoss()

        if self.cuda:
            criterion.cuda()

        return criterion()

    def generalizedDiceLoss(self,
                            pred: Tensor,
                            label: Tensor,
                            reduction: str = 'mean') -> Tensor:
        class GDL(nn.Module):
            def __init__(self) -> None:
                super(GDL, self).__init__()
                self.reduction = reduction

            def forward(self,
                        predict: Tensor = pred,
                        gt: Tensor = label) -> Tensor:
                gt = F.one_hot(gt).permute(0, 3, 1, 2).type_as(predict)
                numerator = torch.sum(predict * gt, dim=[2, 3])
                denominator = torch.sum(predict, dim=[2, 3]) + torch.sum(gt, dim=[2, 3])
                w = 1 / (torch.sum(gt, dim=[2, 3]) ** 2)
                loss = 1 - 2 * torch.sum(w * numerator, dim=1) / torch.sum(w * denominator, dim=1)
                if self.reduction == 'mean':
                    loss = torch.mean(loss, dim=0)
                elif self.reduction == 'sum':
                    loss = torch.sum(loss, dim=0)
                elif self.reduction == 'none':
                    pass
                else:
                    raise ValueError("reduction must be 'mean','sum' or 'none'")
                return loss

        criterion = GDL()

        if self.cuda:
            criterion.cuda()

        return criterion()


if __name__ == '__main__':
    lll = SegmentationLosses(cuda=True)
    a = torch.rand(2, 3, 7, 7).cuda()
    b = torch.randint(0, 3, (2, 7, 7)).cuda()
    print(lll(a, b))
    print(lll(a, b, mode='focal', classNum=3))
    print(lll(a, b, mode='dice'))
    print(lll(a, b, mode='gdl'))
