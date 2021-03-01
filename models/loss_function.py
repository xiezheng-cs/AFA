#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/6 10:30
# @Author  : xiezheng
# @Site    : 
# @File    : loss_function.py


import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_label_smoothing(outputs, labels):
    """
    loss function for label smoothing regularization
    """
    alpha = 0.1
    N = outputs.size(0)  # batch_size
    C = outputs.size(1)  # number of classes
    smoothed_labels = torch.full(size=(N, C), fill_value= alpha / (C - 1), device=outputs.device)
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1-alpha)

    log_prob = torch.nn.functional.log_softmax(outputs, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels) / N

    return loss


def loss_kd_regularization(outputs, labels, reg_alpha=0.10, reg_temperature=20, multiplier=1000):
    """
    loss function for mannually-designed regularization: Tf-KD_{reg}
    """
    alpha = reg_alpha
    T = reg_temperature
    # print('reg_alpha={},reg_temperature={},multiplier={}'.format(reg_alpha, reg_temperature, multiplier))

    correct_prob = 0.99    # the probability for correct class in u(k)
    loss_CE = F.cross_entropy(outputs, labels)
    K = outputs.size(1)

    teacher_soft = torch.ones_like(outputs, device=outputs.device)
    teacher_soft = teacher_soft*(1-correct_prob)/(K-1)  # p^d(k)
    for i in range(outputs.shape[0]):
        teacher_soft[i ,labels[i]] = correct_prob
    loss_soft_regu = nn.KLDivLoss()(F.log_softmax(outputs, dim=1), F.softmax(teacher_soft/T, dim=1))*multiplier
    # loss_soft_regu = nn.KLDivLoss()(F.log_softmax(outputs, dim=1), F.softmax(teacher_soft / T, dim=1))
    # print('loss_CE={}, loss_soft_regu={}'.format(loss_CE, loss_soft_regu))

    KD_loss = (1. - alpha)*loss_CE + alpha*loss_soft_regu

    return KD_loss


def get_loss_type(loss_type, logger=None):

    if loss_type == 'loss_label_smoothing':
        loss_fn = loss_label_smoothing
    elif loss_type == 'loss_kd_regularization':
        loss_fn = loss_kd_regularization

    elif loss_type == 'CrossEntropyLoss':
        loss_fn =  nn.CrossEntropyLoss().cuda()
    else:
        assert False, logger.info("invalid loss_type={}".format(loss_type))

    if logger is not None:
        logger.info("loss_type={}, {}".format(loss_type, loss_fn))
    return loss_fn



if __name__ == '__main__':
    print()