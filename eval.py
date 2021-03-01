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



def eval_net(args, logger):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # data_loader
    train_loader, val_loader, target_class_num, dataset_sizes = \
        get_target_dataloader(args.target_dataset, args.batch_size, args.num_workers, args.target_data_dir,
                              image_size=args.image_size, data_aug=args.data_aug, logger=logger)

    # model setting
    model_source, model_target = get_model(args.base_model_name, args.base_task, target_class_num, logger,
                                           pretrained_path=args.load_pretrain_path)

    # loss
    loss_fn = get_loss_type(loss_type=args.loss_type, logger=logger)

    # optimizer and lr_scheduler
    optimizer, lr_scheduler = None, None
    channel_weights, num_epochs = None, None
    writer, feature_criterions = None, None
    # init framework
    framework = TransferFramework(args, train_loader, val_loader, target_class_num, args.data_aug, args.base_model_name,
                                  model_source, model_target, feature_criterions, loss_fn, args.reg_type,
                                  channel_weights, num_epochs, args.alpha, args.beta, optimizer, lr_scheduler,
                                  writer, logger, print_freq=args.print_freq)

    # val
    val_top1_acc = framework.eval()
    
    logger.info('||==>Val acc={:.6f}\n'.format(val_top1_acc))
    logger.info('experiment_id: {}'.format(args.exp_id))
    return val_top1_acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer')
    parser.add_argument('conf_path', type=str, metavar='conf_path',
                        help='the path of config file for training (default: 64)')
    argparses = parser.parse_args()
    args = Option(argparses.conf_path)
    args.set_save_path()

    # args = parse_args()
    logger = None
    temp = args.outpath
    output_process(args.outpath)
    write_settings(args)
    logger = get_logger(args.outpath, 'attention_transfer_0')
    val_acc = eval_net(args, logger)
