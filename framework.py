import numpy as np
import torch
from torch import nn
from utils.util import AverageMeter
# from prefetch_generator import BackgroundGenerator
from utils.util import get_learning_rate, accuracy, record_epoch_learn_alpha, get_fc_name
from models.regularizer import reg_classifier, reg_fea_map, reg_att_fea_map, reg_l2sp, \
    reg_pixel_att_fea_map_learn, reg_channel_att_fea_map_learn, reg_model


class TransferFramework:

    def __init__(self, args, train_loader, val_loader, target_class_num, data_aug, base_model_name, model_source,
                 model_target, feature_criterions, loss_fn, reg_type, channel_weights, num_epochs, alpha, beta,
                 optimizer, lr_scheduler, writer, logger, print_freq=10):

        self.setting = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.target_class_num = target_class_num
        self.data_aug = data_aug

        self.base_model_name = base_model_name
        self.model_source = model_source
        self.model_target = model_target

        self.model_source_weights = {}
        self.model_target_weights = {}

        self.loss_fn = loss_fn
        self.reg_type = reg_type
        self.feature_criterions = feature_criterions
        self.alpha = alpha
        self.beta = beta
        self.channel_weights = channel_weights

        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.lr = 0.0
        self.lr_scheduler = lr_scheduler
        self.writer = writer
        self.logger = logger
        self.print_freq = print_freq

        # framework init
        self.fc_name = get_fc_name(self.base_model_name, self.logger)
        self.hook_layers = []
        self.layer_outputs_source = []
        self.layer_outputs_target = []
        self.framework_init()

    def framework_init(self):
        if 'fea_map' in self.reg_type:
            self.hook_setting()

        elif self.reg_type in ['l2sp']:
            for name, param in self.model_source.named_parameters():
                if not name.startswith(self.fc_name):
                    self.model_source_weights[name] = param.detach()
                    # print('name={}'.format(name))

        elif self.reg_type in ['l2fe']:
            for name, param in self.model_target.named_parameters():
                if not name.startswith(self.fc_name):
                    param.requires_grad = False

        self.logger.info('self.model_source_weights len = {} !'.format(len(self.model_source_weights)))

    # hook
    def _for_hook_source(self, module, input, output):
        self.layer_outputs_source.append(output)

    def _for_hook_target(self, module, input, output):
        self.layer_outputs_target.append(output)

    def register_hook(self, model, func):
        for name, layer in model.named_modules():
            if name in self.hook_layers:
                layer.register_forward_hook(func)

    def get_hook_layers(self):
        if self.base_model_name == 'resnet101':
            self.hook_layers = ['layer1.2.conv3', 'layer2.3.conv3', 'layer3.22.conv3', 'layer4.2.conv3']

        elif self.base_model_name == 'resnet50':
            self.hook_layers = ['layer1.2.conv3', 'layer2.3.conv3', 'layer3.5.conv3', 'layer4.2.conv3']

        elif self.base_model_name == 'inception_v3':
            self.hook_layers = ['Conv2d_4a_3x3', 'Mixed_5d', 'Mixed_6e', 'Mixed_7c']

        elif self.base_model_name == 'mobilenet_v2':
            self.hook_layers = ['features.5.conv.2', 'features.9.conv.2', 'features.13.conv.2', 'features.17.conv.2']
        else:
            assert False, self.logger.info("invalid base_model_name={}".format(self.base_model_name))

    def hook_setting(self):
        # hook
        self.get_hook_layers()
        self.register_hook(self.model_source, self._for_hook_source)
        self.register_hook(self.model_target, self._for_hook_target)
        self.logger.info("self.hook_layers={}".format(self.hook_layers))

    def train(self, epoch):
        # train mode
        self.model_target.train()
        self.model_source.eval()

        clc_losses = AverageMeter()
        classifier_losses = AverageMeter()
        model_losses = AverageMeter()
        feature_losses = AverageMeter()
        # attention_losses = AverageMeter()
        total_losses = AverageMeter()
        train_top1_accs = AverageMeter()

        self.lr_scheduler.step(epoch)
        self.lr = get_learning_rate(self.optimizer)
        self.logger.info('self.optimizer={}'.format(self.optimizer))

        self.logger.info('feature_loss alpha={}'.format(self.alpha))
        self.logger.info('self.reg_type={}'.format(self.reg_type))

        for i, (imgs, labels) in enumerate(self.train_loader):
            # target_data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()

            # taget forward and loss
            if self.base_model_name == 'inception_v3':
                outputs, _ = self.model_target(imgs)
            else:
                outputs = self.model_target(imgs)

            clc_loss = self.loss_fn(outputs, labels)
            classifier_loss = 0
            feature_loss = 0
            model_loss = 0

            # source_model forward for hook
            if self.reg_type not in ['l2fe', 'l2', 'l2sp']:
                with torch.no_grad():
                    _ = self.model_source(imgs)

            if self.reg_type not in ['l2', 'l2fe']:
                classifier_loss = reg_classifier(self.model_target, self.fc_name)

            if self.reg_type == 'l2sp':
                feature_loss = reg_l2sp(self.model_target, self.fc_name, self.model_source_weights)

            elif self.reg_type == 'fea_map':
                feature_loss = reg_fea_map(self.layer_outputs_source, self.layer_outputs_target)
            elif self.reg_type == 'att_fea_map':
                feature_loss = reg_att_fea_map(self.layer_outputs_source,
                                               self.layer_outputs_target, self.channel_weights)

            # combine loss
            elif self.reg_type == 'pixel_att_fea_map_learn':
                feature_loss = reg_pixel_att_fea_map_learn(self.layer_outputs_source,
                                                           self.layer_outputs_target, self.feature_criterions)
            
            elif self.reg_type == 'channel_att_fea_map_learn':
                feature_loss = reg_channel_att_fea_map_learn(self.layer_outputs_source,
                                                             self.layer_outputs_target, self.feature_criterions)

            if self.reg_type not in ['l2fe', 'l2']:
                total_loss = clc_loss + self.alpha * feature_loss + self.beta * classifier_loss
            else:
                total_loss = clc_loss
                classifier_loss = 0
                feature_loss = 0

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # batch update
            self.layer_outputs_source.clear()
            self.layer_outputs_target.clear()

            clc_losses.update(clc_loss.item(), imgs.size(0))

            if classifier_loss == 0:
                classifier_losses.update(classifier_loss, imgs.size(0))
            else:
                classifier_losses.update(classifier_loss.item(), imgs.size(0))

            if model_loss == 0:
                model_losses.update(model_loss, imgs.size(0))
            else:
                model_losses.update(model_loss.item(), imgs.size(0))

            if feature_loss == 0:
                feature_losses.update(feature_loss, imgs.size(0))
            else:
                feature_losses.update(feature_loss.item(), imgs.size(0))

            total_losses.update(total_loss.item(), imgs.size(0))

            # compute accuracy
            top1_accuracy = accuracy(outputs, labels, 1)
            train_top1_accs.update(top1_accuracy, imgs.size(0))

            # Print status
            if i % self.print_freq == 0:
                self.logger.info(
                    'Train Epoch: [{:d}/{:d}][{:d}/{:d}]\tlr={:.6f}\tclc_loss={:.4f}\t\tclassifier_loss={:.4f}'
                    '\t\tfeature_loss={:.6f}\t\ttotal_loss={:.4f}\ttop1_Accuracy={:.4f}'
                        .format(epoch, self.num_epochs, i, len(self.train_loader), self.lr, clc_losses.avg,
                                classifier_losses.avg, feature_losses.avg, total_losses.avg, train_top1_accs.avg))

        # save tensorboard
        self.writer.add_scalar('lr', self.lr, epoch)
        self.writer.add_scalar('Train_classification_loss', clc_losses.avg, epoch)
        self.writer.add_scalar('Train_classifier_loss', classifier_losses.avg, epoch)
        self.writer.add_scalar('Train_feature_loss', feature_losses.avg, epoch)
        self.writer.add_scalar('Train_total_loss', total_losses.avg, epoch)
        self.writer.add_scalar('Train_top1_accuracy', train_top1_accs.avg, epoch)

        self.logger.info(
            '||==> Train Epoch: [{:d}/{:d}]\tTrain: lr={:.6f}\tclc_loss={:.4f}\t\tclassifier_loss={:.4f}'
            '\t\tfeature_loss={:.6f}\t\ttotal_loss={:.4f}\ttop1_Accuracy={:.4f}'
                .format(epoch, self.num_epochs, self.lr, clc_losses.avg, classifier_losses.avg,
                        feature_losses.avg, total_losses.avg, train_top1_accs.avg))

        return clc_losses.avg, classifier_losses.avg, feature_losses.avg, \
               total_losses.avg, train_top1_accs.avg
    
    def val(self, epoch):
        # test mode
        self.model_target.eval()

        val_losses = AverageMeter()
        fea_losses = AverageMeter()
        val_total_losses = AverageMeter()
        val_classifier_losses = AverageMeter()
        val_top1_accs = AverageMeter()

        # Batches
        for i, (imgs, labels) in enumerate(self.val_loader):
            # Move to GPU, if available
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()

            if self.data_aug == 'improved':
                bs, ncrops, c, h, w = imgs.size()
                imgs = imgs.view(-1, c, h, w)

            # forward and loss
            with torch.no_grad():
                outputs = self.model_target(imgs)
                if self.data_aug == 'improved':
                    outputs = outputs.view(bs, ncrops, -1).mean(1)

                # new add
                _ = self.model_source(imgs)
                if self.data_aug == 'improved':
                    outputs = outputs.view(bs, ncrops, -1).mean(1)

                if self.reg_type not in ['l2', 'l2fe']:
                    classifier_loss = reg_classifier(self.model_target, self.fc_name)

                val_loss = self.loss_fn(outputs, labels)
                if self.reg_type == 'channel_att_fea_map_learn':
                    feature_loss = reg_channel_att_fea_map_learn(self.layer_outputs_source,
                                                                 self.layer_outputs_target, self.feature_criterions)
                
                elif self.reg_type == 'pixel_att_fea_map_learn':
                    feature_loss = reg_pixel_att_fea_map_learn(self.layer_outputs_source, self.layer_outputs_target, 
                                                               self.feature_criterions)

                val_loss = self.loss_fn(outputs, labels)

            # new add 
            val_losses.update(val_loss.item(), imgs.size(0))
            fea_losses.update(feature_loss.item() * self.alpha, imgs.size(0))
            val_classifier_losses.update(classifier_loss.item() * self.beta, imgs.size(0))
            val_total_losses.update(val_loss.item() + feature_loss.item()*self.alpha + self.beta*classifier_loss, imgs.size(0))
            # compute accuracy
            top1_accuracy = accuracy(outputs, labels, 1)
            val_top1_accs.update(top1_accuracy, imgs.size(0))

            # batch update
            self.layer_outputs_source.clear()
            self.layer_outputs_target.clear()

            # new add
            # Print status
            if i % self.print_freq == 0:
                self.logger.info('Val Epoch: [{:d}/{:d}][{:d}/{:d}]\tval_loss={:.4f}\t feature_loss={:4f}\t classifier_loss={:4f}\t total_loss={:4f}\ttop1_accuracy={:.4f}\t'
                                 .format(epoch, self.num_epochs, i, len(self.val_loader), val_losses.avg, fea_losses.avg, val_classifier_losses.avg, val_total_losses.avg,
                                         val_top1_accs.avg))

        # new add 
        self.writer.add_scalar('Test_epoch_loss', val_total_losses.avg, epoch)
        self.writer.add_scalar('Test_epoch_fea_loss', fea_losses.avg, epoch)
        self.writer.add_scalar('Test_epoch_ce_loss', val_losses.avg, epoch)
        self.writer.add_scalar('Test_epoch_acc', val_top1_accs.avg, epoch)

        self.logger.info('||==> Val Epoch: [{:d}/{:d}]\tval_loss={:.4f}\tfea_loss={:4f}\tclassifier_loss={:4f}\t total_loss={:4f}\ttop1_accuracy={:.4f}'
                         .format(epoch, self.num_epochs, val_losses.avg, fea_losses.avg, val_classifier_losses.avg, val_total_losses.avg, val_top1_accs.avg))


        return val_losses.avg, val_top1_accs.avg

        def eval(self):
        val_top1_accs = AverageMeter()
        # test mode
        self.model_target.eval()
        # Batches
        for i, (imgs, labels) in enumerate(self.val_loader):
            # Move to GPU, if available
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()

            if self.data_aug == 'improved':
                bs, ncrops, c, h, w = imgs.size()
                imgs = imgs.view(-1, c, h, w)

            # forward and loss
            with torch.no_grad():
                outputs = self.model_target(imgs)
                if self.data_aug == 'improved':
                    outputs = outputs.view(bs, ncrops, -1).mean(1)

                val_loss = self.loss_fn(outputs, labels)

            top1_accuracy = accuracy(outputs, labels, 1)
            val_top1_accs.update(top1_accuracy, imgs.size(0))

    
            if i % self.print_freq == 0:
                self.logger.info('top1_accuracy={:.6f}' .format(val_top1_accs.avg))
        return val_top1_accs.avg
