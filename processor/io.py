import sys 
import argparse
import yaml
import numpy as np

import torch 
import torch.nn as nn

import torchlight

class IO():
    def __init__(self, argv=None):
        
        self.load_arg(argv)
        
    def load_arg(self, argv=None):
        pass
    
    def init_environment(self):
        pass
    
    def load_model(self):
        pass
    
    def load_weights(self):
        pass
    
    def gpu(self):
        pass
    
    def start(self):
        pass
    
    def start(self):
        pass
    
    @staticmethod
    def get_paser(add_help=False):
        
        parser = argparse.ArgumentParser(add_help=add_help, description='IO Processor')
        