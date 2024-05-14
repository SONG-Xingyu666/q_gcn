import argparse
import os 
import sys
import traceback
import time 
import warnings
import pickle
from collections import OrderedDict
import yaml
import numpy as np
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
    
class IO():
    def __init__(self, work_dir, save_log=True, print_log=True):
        self.work_dir = work_dir
        self.save_log = save_log
        self.print_to_screen = print_log
        self.cur_time = time.time()
        self.split_timer = {}
        self.pavi_logger = None
        self.session_file = None
        self.model_text = ''
        
        def log(self, *args, **kwargs):
            pass
        
        def load_model(self, model, **model_args):
            Model = import_class(model)
        
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolen value expected')
    
def str2dict(v):
    return eval('dict{}'.format(v))

        
def import_class(import_str):
    mod_str, _sep, class_str = import_str.partition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                          (class_str, 
                           traceback.format_exception(*sys.exc_info())))