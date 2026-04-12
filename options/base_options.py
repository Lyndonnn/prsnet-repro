### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import argparse
import os
from utils import util
import torch

def parse_max_dataset_size(value):
    if isinstance(value, float):
        return value
    text = str(value).lower()
    if text in ('inf', 'infinity', 'none', 'all'):
        return float("inf")
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError('max_dataset_size must be non-negative, or inf/all')
    return parsed

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.parser.add_argument('--name', type=str, default=os.environ.get('PRSNET_EXP_NAME', 'exp'), help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default=os.environ.get('PRSNET_GPU_IDS', '0'), help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default=os.environ.get('PRSNET_CHECKPOINTS_DIR', './checkpoints'), help='models are saved here')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        self.parser.add_argument('--activation', type=str, default='lrelu', help='activation type')
        self.parser.add_argument('--bn', action='store_true', default=False, help='whether using batch normalization layers')

        self.parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default=os.environ.get('PRSNET_DATAROOT', './datasets/shapenet/'))
        self.parser.add_argument('--noshuffle', action='store_true', help='if true, takes data in order to make batches, otherwise takes them randomly') 
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=parse_max_dataset_size, default=float("inf"), help='Maximum number of samples allowed per dataset. Accepts an integer or inf/all.')

        # for displays
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
        
        self.parser.add_argument('--gridBound', type=float, default=0.5, help='# of grid bound')
        self.parser.add_argument('--gridSize', type=int, default=32, help='# of grid size')
        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input channels')
        self.parser.add_argument('--output_nc', type=int, default=4, help='# of input channels in first conv layer')
        self.parser.add_argument('--conv_layers', type=int, default=5, help='# of conv layers')
        self.parser.add_argument('--num_plane', type=int, default=3, help='# of symmetry planes')
        self.parser.add_argument('--num_quat', type=int, default=3, help='# of quats')

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            if torch.cuda.is_available():
                torch.cuda.set_device(self.opt.gpu_ids[0])
            else:
                print('[Options] CUDA gpu_ids were requested, but torch.cuda.is_available() is false. Falling back to CPU.')
                self.opt.gpu_ids = []

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
