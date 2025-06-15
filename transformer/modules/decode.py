import torch
from torch import nn
from torch import Tensor
from modules.attention import MultiHeadAttention
from typing import Optional

class DecoderBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_hidden_dim: int,
                 dropout: float
                 ):
        """Initialize Decoder Class Variables

        Args:
            embed_dim: embedding dimension
            num_heads: number of decoders in the stack
            ffn_hidden_dim: hidden FFN layer size
            dropout: dropout rate
        """
        super().__init__()

        # Attention Layers
        self.masked_self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads)

        # Normalization Layers
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.norm_3 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, embed_dim)
        )

        self.dropout = dropout

    def forward(self,
                x: Tensor,
                encoder_output: Tensor,
                target_mask: Optional[Tensor] = None,
                encoder_mask: Optional[Tensor] = None
                ) -> Tensor:
        """Move input through the decoder block
        
        Args:
            x: input data (shape=(B, L, D))
            encoder_output: output from the encoder
            target_mask: prevents looking ahead at future token vectors
            encoder_mask: prevents padded positions

        Returns: decoder block output
        """
        # Masked attention (sub-layer 1)
        masked_attention = self.masked_self_attention(x, x, x, target_mask)
        x = self.norm_1(x + self.dropout(masked_attention))

        # Cross attention (sub-layer 2)
        cross_attention = self.cross_attention(x, encoder_output, encoder_output, encoder_mask)
        x = self.norm_2(x + self.dropout(cross_attention))

        # Feed forward network
        ffn_output = self.ffn(x)
        x = self.norm_3(x + ffn_output)

        return x

class Decoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 max_len: int,
                 N: int,
                 embed_dim: int,
                 num_heads: int,
                 ffn_hidden_dim: int,
                 dropout: float
                 ):
        """Initialize Decoder Class Variables

        Args:
            vocab_size: number of distinct tokens in the vocab
            max_len: maximum sequence length the transformer can handle
            N: number of encoders and decoders
            embed_dim: embedding dimension
            num_heads: number of scaled dot-product attention heads in multi-head attention
            ffn_hidden_dim: hidden layer dimension
            dropout: dropout rate
        """
        super().__init__()

        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ffn_hidden_dim, dropout)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: Tensor,
                encoder_output: Tensor,
                target_mask: Optional[Tensor],
                encoder_mask: Optional[Tensor] = None
                ) -> Tensor:
        """Pass input through the decoder stack

        Args:
            x: input data (shape=(B, L, D))
            encoder_output: output from the encoder
            target_mask: prevents looking ahead at future token vectors
            encoder_mask: prevents padded positions

        Returns: decoder output
        """
        B, L = x.shape
        positions = torch.arange(0, L, device=x.device).unsqueeze(0).expand(B, L)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, encoder_output, target_mask, encoder_mask)
        
        return x