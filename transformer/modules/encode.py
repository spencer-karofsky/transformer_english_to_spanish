import torch
from torch import nn
from torch import Tensor
from .attention import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_hidden_dim: int,
                 dropout: float
                 ):
        """Initialize Encoder Block Class Variables
        
        Args:
            embed_dim: embedding dimension (length of each token vector in the sequence)
            num_heads: number of self-attention heads used for multi-head attention
            ffn_hidden_dim: hidden layer dimension
            dropout: dropout rate
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, embed_dim)
        )
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Self-attention + residual + norm
        
        Args:
            x: Query, Key, Value
            mask: Attention mask of shape (B, h, L, L) where masked positions are 0 and unmasked are 1
        
        Returns: output from encoder block
        """
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm_1(x + self.dropout(attention_output))

        ffn_output = self.ffn(x)
        x = self.norm_2(x + self.dropout(ffn_output))
        
        return x

class Encoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 max_len: int,
                 N: int = 6,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 ffn_hidden_dim: int = 2048,
                 dropout: float = .1):
        """Initialize Encoder Class Variables

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

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        # Encoder stack
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ffn_hidden_dim, dropout)
            for _ in range(N)
        ])

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Pass input through the encoder stack

        Args:
            x: Query, Key, Value (shape=(B, L))
            mask: Attention mask of shape (B, h, L, L) where masked positions are 0 and unmasked are 1

        Returns: encoder output
        """
        B, L = x.shape

        # Token and positional embeddings
        positions = torch.arange(0, L, device=x.device).unsqueeze(0).expand(B, L)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        # Pass encoded and embedded input through the encoder stack
        for layer in self.layers:
            x = layer(x, mask)

        return x