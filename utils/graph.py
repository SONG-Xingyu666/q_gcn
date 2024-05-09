import numpy as np

class Graph():
    """The graph to model the human skeleton
    """
    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        """

        Args:
            layout (str, optional): the layout for 2D pose detector. Defaults to 'openpose'.
            strategy (str, optional): the sampling strategy. Defaults to 'uniform'.
            max_hop (int, optional): the max distance between two connected nodes. Defaults to 1.
            dilation (int, optional): controls the spacing between the kernel points. Defaults to 1.
        """
        self.max_hop = max_hop
        self.dilation = dilation
        
        self.get_edge(layout)
        self.hop_dis = self.get_hop_distance()
        
    def get_edge(self, layout):
        pass
    
    def get_hop_distance(self):
        pass
    
    def get_adjacency(self):
        pass