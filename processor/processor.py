import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .io import IO

class Processor(IO):
    def __init__(self, argv=None):
        
        pass
    
    def init_environment(self):
        return super().init_environment()
    
    def load_optimizer(self):
        pass
    
    def load_data(self):
        pass
    
    def show_wpoch_info(self):
        pass
    
    def show_iter_info(self):
        pass
    
    def train(self):
        pass
    
    def test(self):
        pass
    
    def start(self):
        return super().start()
    
    @staticmethod
    def get_paser(add_help=False):
        
        parser = argparse.ArgumentParser(add_help=add_help, description='Base Processor')
        
        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='the path to the configuration file')

        # Processor 
        parser.add_argument('--phase', default='./work_dir/tmp', help='the roek folder for storing results')
        parser.add_argument('--save_result', type=str2bool, default=False, help='if true, the ourput of model will be saved')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='the indexes of GPUs for training or testing')
        
        # Visulize and debug
        parser.add_argument('--log_interval', type=int, default=100, help='the interval for pring messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=10, help='the interval for storing models (#iteration)')
        parser.add_argument('--eval_interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
        parser.add_argument('--pavi_log', type=str2bool, default=False, help='logging on pavi or not')
        
        # feeder
        parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
        parser.add_argument('--num_worker', type=int, default=4, help='the number of worker per gpu for data loader')
        parser.add_argument('--train_feeder_args', action=DictAction, default=dict(), help='the arguments of dataloader for training')
        parser.add_argument('--test_feeder_args', action=DictAction, default=dict(), help='the arguments of dataloader for testing')
        parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
        parser.add_argument('--debug', action='store_true', help='less data, faster loading')
        
        # model
        parser.add_argument('--model', default=None, help='the model will be used')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--weights', default=None, help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
        
        return parser