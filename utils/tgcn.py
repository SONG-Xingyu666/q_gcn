import torch
import torch.nn as nn

class TemporalGraphConvolution(nn.Module):
    """Basic module for applying a graph convolution

    Args:
        nn (_type_): _description_
    """
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size,
                 t_kernel_size=1, 
                 t_stride=1, 
                 t_padding=1,
                 t_dilation=1, 
                 bias=True):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels*self.kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding,0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias
        )
        
    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        
        x = self.conv(x)
        
        n, kc, t, v = x.size()
        
        
        