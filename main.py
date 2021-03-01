import os
import argparse
import json
import random

import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from option import Option
from framework import TransferFramework
from models.loss_function import get_loss_type
from data.dataloader import get_target_dataloader
from models.get_model import get_model
from utils.checkpoint import save_checkpoint, save_model
from utils.util import get_logger, output_process, record_epoch_data, \
    write_settings, get_optimier_and_scheduler, get_channel_weight
from models.regularizer import get_reg_criterions



def train_net(args, logger, seed):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    logger.info("seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # cudnn
    writer = SummaryWriter(args.outpath)

    start_epoch = 0
    val_best_acc = 0
    val_best_acc_index = 0
    feature_criterions = None

    # data_loader
    train_loader, val_loader, target_class_num, dataset_sizes = \
        get_target_dataloader(args.target_dataset, args.batch_size, args.num_workers, args.target_data_dir,
                              image_size=args.image_size, data_aug=args.data_aug, logger=logger)

    # model setting
    model_source, model_target = get_model(args.base_model_name, args.base_task, target_class_num, logger,
                                           pretrained_path=args.load_pretrain_path)

    # get channel_weights
    channel_weights = get_channel_weight(args.channel_weight_path, logger)

    # get feature_criterions
    if args.reg_type in ['pixel_att_fea_map_learn', 'channel_att_fea_map_learn', 'channel_pixel_att_fea_map_learn',
                         'channel_att_fea_map_without_params', 'pixel_att_fea_map_without_params']:
        feature_criterions = get_reg_criterions(args, logger)

    # iterations -> epochs
    if args.reg_type != 'l2fe':
        num_epochs = int(np.round(args.max_iter * args.batch_size / dataset_sizes))
        step = [int(0.67 * num_epochs)]
        logger.info('num_epochs={}, step={}'.format(num_epochs, step))
    else:
        num_epochs = 10
        step = 6

    # loss
    loss_fn = get_loss_type(loss_type=args.loss_type, logger=logger)

    # optimizer and lr_scheduler
    optimizer, lr_scheduler = get_optimier_and_scheduler(args, model_target, feature_criterions, step, logger)

    # init framework
    framework = TransferFramework(args, train_loader, val_loader, target_class_num, args.data_aug, args.base_model_name,
                                  model_source, model_target, feature_criterions, loss_fn, args.reg_type,
                                  channel_weights, num_epochs, args.alpha, args.beta, optimizer, lr_scheduler,
                                  writer, logger, print_freq=args.print_freq)

    # Epochs
    for epoch in range(start_epoch, num_epochs):
        # train epoch
        clc_loss, classifier_loss, feature_loss, train_total_loss, train_top1_acc = framework.train(epoch)
        # val epoch
        val_loss, val_top1_acc = framework.val(epoch)
        # record into txt
        record_epoch_data(args.outpath, epoch, clc_loss, classifier_loss, feature_loss, train_total_loss,
                          train_top1_acc, val_loss, val_top1_acc)

        if val_top1_acc > val_best_acc:
            val_best_acc = val_top1_acc
            val_best_acc_index = epoch
            # save model_target
            save_model(args.outpath, val_best_acc_index, model_target, val_best_acc)

        logger.info('||==>Val Epoch: Val_best_acc_index={}\tVal_best_acc={:.4f}\n'.format(val_best_acc_index, val_best_acc))
        logger.info('experiment_id: {}'.format(args.exp_id))
    return val_best_acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer')
    parser.add_argument('conf_path', type=str, metavar='conf_path',
                        help='the path of config file for training (default: 64)')
    argparses = parser.parse_args()
    args = Option(argparses.conf_path)
    args.set_save_path()

    # args = parse_args()
    best_val_acc_list = []
    logger = None
    temp = args.outpath
    for i in range(1, args.repeat+1):
        if args.repeat != 1:
            args.outpath = temp + "_{:02d}".format(i)

        output_process(args.outpath)
        write_settings(args)
        logger = get_logger(args.outpath, 'attention_transfer_{:02d}'.format(i))
        val_acc = train_net(args, logger, args.seed + i - 1)
        best_val_acc_list.append(val_acc)

    acc_mean = np.mean(best_val_acc_list)
    acc_std = np.std(best_val_acc_list)
    for i in range(len(best_val_acc_list)):
        print_str = 'repeat={}\tbest_val_acc={}'.format(i, best_val_acc_list[i])
        logger.info(print_str)
    logger.info('All repeat val_acc_mean={}\tval_acc_std={})'.format(acc_mean, acc_std))
