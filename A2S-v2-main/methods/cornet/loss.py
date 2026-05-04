import torch
from torch import nn
from torch.nn import functional as F

from util import *


def cos_simi(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)
    return torch.clamp(sim, min=0.0005, max=0.9995)


class SimMinLoss(nn.Module):
    def __init__(self, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.metric = metric
        self.reduction = reduction

    def forward(self, embedded_bg, embedded_fg):
        if self.metric != 'cos':
            raise NotImplementedError
        sim = cos_simi(embedded_bg, embedded_fg)
        loss = -torch.log(1 - sim)
        if self.reduction == 'mean':
            return torch.mean(loss)
        if self.reduction == 'sum':
            return torch.sum(loss)
        return loss


class SimMaxLoss(nn.Module):
    def __init__(self, metric='cos', alpha=0.25, reduction='mean'):
        super(SimMaxLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, embedded_bg):
        if self.metric != 'cos':
            raise NotImplementedError
        sim = cos_simi(embedded_bg, embedded_bg)
        loss = -torch.log(sim)
        loss[loss < 0] = 0
        _, indices = sim.sort(descending=True, dim=1)
        _, rank = indices.sort(dim=1)
        rank = rank - 1
        rank_weights = torch.exp(-rank.float() * self.alpha)
        loss = loss * rank_weights
        if self.reduction == 'mean':
            return torch.mean(loss)
        if self.reduction == 'sum':
            return torch.sum(loss)
        return loss


def ccam_loss(preds, alpha=0.25):
    loss1 = SimMaxLoss(metric='cos', alpha=alpha).to(preds['fg_feats'].device)(preds['fg_feats'])
    loss2 = SimMinLoss(metric='cos').to(preds['fg_feats'].device)(preds['bg_feats'], preds['fg_feats'])
    loss3 = SimMaxLoss(metric='cos', alpha=alpha).to(preds['fg_feats'].device)(preds['bg_feats'])
    return loss1, loss2, loss3, loss1 + loss2 + loss3

def IOU(pred, target):
    inter = target * pred
    union = target + pred - target * pred
    iou_loss = 1 - torch.sum(inter, dim=(1, 2, 3)) / (torch.sum(union, dim=(1, 2, 3)) + 1e-7)
    return iou_loss.mean()

def Loss(preds, target, config):
    loss = 0
    
    for pred in preds['sal']:
        pred = nn.functional.interpolate(pred, size=target.size()[-2:], mode='bilinear')
        target = target.gt(0.5).float()
        loss += IOU(torch.sigmoid(pred), target)
        
    return loss
