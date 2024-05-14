import os
import torch

def visible_gpu(gpus):
    
    gpus = [gpus] if isinstance(gpus, int) else list(gpus)