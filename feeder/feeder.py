import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms

# visualization
import time

# operation
from . import tools

class Feeder(torch.utils.data.Dataset):
    
    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True):
        """Feeder for skeleton-based action recognition

        Args:
            data_path (str): the path to '.npy' data, shape: (N, C, T, V, M)
            label_path (str): the path to labels
            random_choose (bool, optional): if True, randomly choose a portion of the input sequence. Defaults to False.
            random_mode (bool, optional): if True, randomly pad zeros at the begining or end of sequence. Defaults to False.
            window_size (int, optional): the length of the output sequence. Defaults to -1.
            debug (bool, optional): if True, only use the first 100 samples. Defaults to False.
            mmap (bool, optional): _description_. Defaults to True.
        """
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        
        self.load_data(mmap=mmap)
        
        
    def load_data(self, mmap):
        
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
            
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
            
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]
            
        self.N, self.C, self.T, self.V, self.M = self.data.shape
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        
        # precessing
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        
        return data_numpy, label