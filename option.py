import os
import shutil

from pyhocon import ConfigFactory


class Option(object):
    def __init__(self, conf_path):
        super(Option, self).__init__()
        self.conf = ConfigFactory.parse_file(conf_path)


        # ------------- data options -------------------------------------------
        # target_dataset
        self.target_dataset = self.conf['target_dataset']  # path for loading data set
        self.target_data_dir = self.conf['target_data_dir']


        # ------------- general options ----------------------------------------
        self.outpath = self.conf['outpath']  # log path
        self.gpu_id = self.conf['gpu_id']  # GPU id to use, e.g. "0,1,2,3"
        # self.seed = self.conf['seed']  # manually set RNG seed
        self.print_freq = self.conf['print_freq']  # print frequency (default: 10)
        self.batch_size = self.conf['batch_size']  # mini-batch size
        self.num_workers = self.conf['num_workers']  # num_workers
        self.exp_id = self.conf['exp_id']  # identifier for experiment


        # ------------- common optimization options ----------------------------
        self.seed = self.conf['seed']
        self.repeat = self.conf['repeat']
        self.lr = float(self.conf['lr'])  # initial learning rate
        self.max_iter = self.conf['max_iter']  # number of total epochs
        self.momentum = self.conf['momentum']  # momentum
        self.weight_decay = float(self.conf['weight_decay'])  # weight decay
        self.gamma = self.conf['gamma']


        # ------------- model options ------------------------------------------
        self.base_task = self.conf['base_task']
        self.base_model_name = self.conf['base_model_name']
        self.image_size = self.conf['image_size']
        self.data_aug = self.conf['data_aug']


        # ------------- attention transfer options ------------------------------------------
        self.alpha = float(self.conf['alpha'])
        self.beta = self.conf['beta']
        # self.r = self.conf['r']
        # self.attention_gamma = self.conf['attention_gamma']
        self.reg_type = self.conf['reg_type']

        # self.use_channel_attention = self.conf['use_channel_attention']
        # self.use_pixel_attention = self.conf['use_pixel_attention']

        self.loss_type = self.conf['loss_type']
        self.lr_scheduler = self.conf['lr_scheduler']
        self.channel_weight_path = self.conf['channel_weight_path']
        # self.alpha_decay = self.conf['alpha_decay']


        # ---------- resume or pretrained options ---------------------------------
        # path to pretrained model
        self.pretrain_path = None if len(self.conf['pretrain_path']) == 0 else self.conf['pretrain_path']
        # path to directory containing checkpoint
        self.resume = None if len(self.conf['resume']) == 0 else self.conf['resume']
        self.load_pretrain_path = None if len(self.conf['pre_trained_path']) == 0 else self.conf['pre_trained_path']

    def set_save_path(self):
        # exp_id = 'log_{}_{}_img{}_da-{}_{}_iter{}_bs{}_{}_lr{}_wd{}_{}_alpha{}_ad-{}_r-{}_att-gamma{}_{}'\
        #     .format(self.base_task, self.target_dataset, self.image_size, self.data_aug, self.base_model_name,
        #             self.max_iter, self.batch_size, self.lr_scheduler, self.lr, self.weight_decay,
        #             self.reg_type, self.alpha, self.alpha_decay, self.r, self.attention_gamma, self.exp_id)

        exp_id = 'log_{}_{}_img{}_da-{}_{}_iter{}_bs{}_{}_lr{}_wd{}_{}_alpha{}_{}' \
            .format(self.base_task, self.target_dataset, self.image_size, self.data_aug, self.base_model_name,
                    self.max_iter, self.batch_size, self.lr_scheduler, self.lr, self.weight_decay,
                    self.reg_type, self.alpha, self.exp_id)

        path = '{}_{}_da-{}_{}'.format(self.reg_type, self.target_dataset, self.data_aug, self.base_model_name)
        self.outpath = os.path.join(self.outpath, path, exp_id)
        # self.outpath = os.path.join(self.outpath, exp_id)

