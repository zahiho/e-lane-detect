#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn

def cross_entropy2d(logit, target, ignore_index=255, weight=None, reduct='elementwise_mean'):
    n, c, h, w = logit.size()
    #print("Inside Cross entropy ",logit.size()  ,"Target Size",target.size())
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    if weight is None:
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index,reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,reduction='sum')
    loss = criterion(logit, target.long())

    batch_average=True
    if batch_average:
        loss /= n

    return loss
