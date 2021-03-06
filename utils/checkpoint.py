#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/23 22:25
# @Author  : xiezheng
# @Site    : 
# @File    : checkpoint.py

import os

import torch
from torch import nn
from utils.util import ensure_folder


def save_checkpoint(outpath, epoch, model, feature_criterions, optimizer, lr_scheduler, val_best_acc, logger):
    check_point_params = {}
    if isinstance(model, nn.DataParallel):
        check_point_params["model"] = model.module.state_dict()
    else:
        check_point_params["model"] = model.state_dict()

    if feature_criterions:
        if isinstance(feature_criterions, nn.DataParallel):
            check_point_params["feature_criterions"] = feature_criterions.module.state_dict()
        else:
            check_point_params["feature_criterions"] = feature_criterions.state_dict()
    else:
        logger.info("feature_criterions is None, don't save!")

    check_point_params["val_best_acc"] = val_best_acc
    check_point_params["optimizer"] = optimizer.state_dict()
    check_point_params["lr_scheduler"] = lr_scheduler.state_dict()
    check_point_params['epoch'] = epoch
    # logger.info('optimizer={}'.format(optimizer))

    output_path = os.path.join(outpath, "check_point")
    ensure_folder(output_path)
    filename = 'checkpoint.pth'
    torch.save(check_point_params, os.path.join(output_path, filename))


def save_model(outpath, epoch, model, val_best_acc):
    check_point_params = {}
    if isinstance(model, nn.DataParallel):
        check_point_params["model"] = model.module.state_dict()
    else:
        check_point_params["model"] = model.state_dict()

    output_path = os.path.join(outpath, "check_point")
    ensure_folder(output_path)
    filename = 'model_{:03d}_acc{}.pth'.format(epoch, val_best_acc)
    torch.save(check_point_params, os.path.join(output_path, filename))


def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, logger):
    check_point_params = torch.load(checkpoint_path)
    model_state = check_point_params["model"]
    start_epoch = check_point_params['epoch']
    optimizer.load_state_dict(check_point_params["optimizer"])
    lr_scheduler.load_state_dict(check_point_params["lr_scheduler"])
    val_best_acc = check_point_params["val_best_acc"]

    model = load_state(model, model_state, logger)
    return model, start_epoch, optimizer, lr_scheduler, val_best_acc


def load_state(model, state_dict, logger):
    """
    load state_dict to model
    :params model:
    :params state_dict:
    :return: model
    """

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    logger.info("load model state finished !!!")
    return model


def load_pretrain_model(pretrain_path, model, logger):
    if pretrain_path is not None:
        check_point_params = torch.load(pretrain_path)

        model_state = check_point_params['model']
        # model_state = check_point_params

        model = model.load_state(model_state)
        logger.info("|===>load restrain file: {}".format(pretrain_path))
    else:
        logger.info('pretrain_path is None')

    return model
