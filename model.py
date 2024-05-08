import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, vertex_num, ) -> None:
        super().__init__()