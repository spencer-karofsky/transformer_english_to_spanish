"""
Transformer Implementation from 'Attention is All You Need' paper: https://arxiv.org/pdf/1706.03762
"""
from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        """ Initialize Encoder Class Variables
        """
        pass


class Transformer(nn.Module):
    def __init__(self, N: int = 6, num_heads: int = 8):
        """Initialize Transformer Class Variables

        Args:
            N: number of encoders and decoders
            num_heads: number of scaled dot-product attention heads in multi-head attention       
        """
        self.N = N
        self.num_heads = num_heads

        pass
    
    