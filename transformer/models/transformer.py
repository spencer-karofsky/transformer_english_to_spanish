"""
Transformer Implementation from 'Attention is All You Need' paper: https://arxiv.org/pdf/1706.03762
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from encode import Encoder
from decode import Decoder
from typing import Optional


class Transformer(nn.Module):
    def __init__(self,
                 source_vocab_size: int,
                 target_vocab_size: int,
                 max_len: int,
                 N: int = 6,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 ffn_hidden_dim: int = 2048,
                 dropout: float = 0.1
                 ):
        """Initialize Transformer Class Variables

        Args:
            soruce_vocab_size: source vocab size
            target_vocab_size: target vocab size
            max_len: maximum sequence length
            N: number of encoders and decoders
            embed_dim: embedding vector size
            num_heads: number of scaled dot-product attention heads in multi-head attention 
            ffn_hidden_dim: FFN hidden layer number of units
            dropout: dropout rate      
        """
        super().__init__()
        self.N = N
        self.num_heads = num_heads

        self.encoder = Encoder(
            vocab_size=source_vocab_size,
            max_len=max_len,
            N=N,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_hidden_dim=ffn_hidden_dim,
            dropout=dropout
        )

        self.decoder = Decoder(
            target_vocab_size,
            max_len,
            N,
            embed_dim,
            num_heads,
            ffn_hidden_dim,
            dropout
        )

        self.output_projection = nn.Linear(embed_dim, target_vocab_size)

    def forward(self,
                source: Tensor,
                target: Tensor,
                source_mask: Optional[Tensor],
                target_mask: Optional[Tensor],
                encoder_mask: Optional[Tensor]):
        """Pass an input through the transformer
        
        Args:
            source: our source data
            target: our target data
            source_mask: source mask
            target_mask: target mask
            encoder mask: encoder mask
        """
        # Encode
        encoder_output = self.encoder(source, source_mask)

        # Decode
        decoder_output = self.decoder(target, encoder_output, target_mask, encoder_mask)

        # Return output logit probabilities
        logits = self.output_projection(decoder_output)
        
        return logits
    