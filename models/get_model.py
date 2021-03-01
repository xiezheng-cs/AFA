#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/23 20:51
# @Author  : xiezheng
# @Site    : 
# @File    : get_model.py


import pickle
import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101, inception_v3, mobilenet_v2


def pretrained_model_imagenet(base_model):
    return eval(base_model)(pretrained=True)


def pretrained_model_places365(base_model):
    assert base_model == 'resnet50'
    model = resnet50(pretrained=False, num_classes=365)
    state_dict = torch.load('resnet50_places365_python36.pth.tar', pickle_module=pickle)['state_dict']
    state_dict_new = {}
    for k, v in state_dict.items():
        state_dict_new[k[len('module.'):]] = v
    model.load_state_dict(state_dict_new)
    return model


def get_base_model(base_model, base_task):
    if base_task == 'places365':
        return pretrained_model_places365(base_model)
    else:
        return pretrained_model_imagenet(base_model)


def get_model(base_model_name, base_task, target_class_num, logger, pretrained_path=None):
    model_source = get_base_model(base_model_name, base_task)
    model_target = get_base_model(base_model_name, base_task)

    # replace fc
    if 'resnet' in base_model_name:
        model_target.fc = nn.Linear(model_target.fc.in_features, target_class_num)
    elif 'inception' in base_model_name:
        model_target.fc = nn.Linear(model_target.fc.in_features, target_class_num)
    elif 'mobilenet' in base_model_name:
        model_target.classifier = nn.Sequential(list(model_target.classifier.children())[0],
                                                nn.Linear(list(model_target.classifier.children())[1].in_features, target_class_num))
        # model_target.AuxLogits.fc = nn.Linear(model_target.AuxLogits.fc.in_features, target_class_num)

    if pretrained_path is not None:
        model_target.load_state_dict(torch.load(pretrained_path)['model'])
        model_source_dict = model_source.state_dict()
        pre_trained_model_dict = torch.load(pretrained_path)['model']
        pre_trained_dict = {k: v for k, v in pre_trained_model_dict.items() if k in model_source_dict and 'fc' not in k and 'classifier' not in k}
        logger.info("model_source_pre_trained_dict size: {}".format(len(pre_trained_dict)))
        model_source_dict.update(pre_trained_dict)
        model_source.load_state_dict(model_source_dict)
        logger.info("Finish Load State dict!!")

    for param in model_source.parameters():
        param.requires_grad = False
    model_source.eval()

    model_target = model_target.cuda()
    model_source = model_source.cuda()
    logger.info('base_task = {}, get model_source and model_target = {}'.format(base_task, base_model_name))
    # assert False
    return model_source, model_target


if __name__ == '__main__':
    # model = inception_v3(pretrained=False, aux_logits=False)
    model = resnet101(pretrained=False)

    count = 1
    for name, param in model.named_parameters():
        print('count={}, name={}'.format(count, name))
        # if 'conv' in name or 'downsample.0' in name:
        #     print('count={}\tname={}\tsize={}'.format(count, name,
        #                                                      param.shape[0]*param.shape[1]*param.shape[2]*param.shape[3]))
        count += 1
