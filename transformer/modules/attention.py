"""
From 'Attention is All You Need' Paper

Three types of Self-Attention Used in the Paper:
    1) Self-Attention: used in the encoder; every token attends to every other token
    2) Masked Self-Attention: used in the decoder: same as self-attention, but doesn't attend to future tokens
    3) Cross Attention: used in decoder: attends to encoder's output
"""
from typing import Optional
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                Q: Tensor,
                K: Tensor,
                V: Tensor,
                mask: Tensor | None = None,
                ) -> Tensor:
        """Computes Scaled Dot Product Attention
        Formula: Attention=Softmax((QK^T)/(sqrt(d_k)))V
        Meaning: How much should I focus on each vector (in Values tensor) when creating a new output vector?
        
        Args:
            Q: Query Tensor– our search query (shape=(B,h,L,d_k))
            K: Keys Tensor– what each vector advertises (shape=(B,h,L,d_k))
            V: Values Tensor– what each vector stores (shape=(B,h,L,d_v))
            mask: Attention mask of shape (B, h, L, L) where masked positions are 0 and unmasked are 1
        Returns:
            attention: Output attention-weighted values of shape (B, h, L, d_v)
        Raises:
            AssertationError: If Q, K, or V have mismatched shapes.
        """
        # Validate dimensionality compatability
        assert Q.shape == K.shape, f'Q ({Q.shape}) and K ({K.shape}) have mismatched dimensions'
        assert Q.shape[:3] == V.shape[:3], f'Q & K ({Q.shape}) and V ({V.shape}) have mismatched dimensions in the first three dimensions'

        # Transpose K, matrix-multiply with Q, and scale
        d_k = Q.shape[-1]
        attention_scores = Q @ K.transpose(-2, -1)
        attention_scores /= math.sqrt(d_k)

        # Mask attention scores if enabled
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Take Softmax and multiply with V
        attention = F.softmax(attention_scores, dim=-1) # apply Softmax over keys (dim=-1)
        attention = attention @ V

        return attention
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        """In practice, we compute smaller attentions in parallel and combine results for better performance
        
        Args:
            embed_dim: embedding dimension (length of each token vector in the sequence)
            num_heads: number of self-attention heads used for multi-head attention
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads

        # Linear projections
        self.Q_projection = nn.Linear(embed_dim, embed_dim)
        self.K_projection = nn.Linear(embed_dim, embed_dim)
        self.V_projection = nn.Linear(embed_dim, embed_dim)

        self.output_projection = nn.Linear(embed_dim, embed_dim)

        self.self_attention = SelfAttention()

    def _split_heads(self, x: Tensor) -> Tensor:
        """Splits data into heads

        (B, L, embed_dim) -> (B, h, L, head_dim)

        Args:
            x: the input data (already embedded and positionally encoded)
        
        Returns: x split into heads
        """
        B, L, _ = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.contiguous().transpose(1, 2)
    
    def _combine_heads(self, x: Tensor) -> Tensor:
        """Combines attention head outputs
        
        (B, num_heads, L, head_dim) -> (B, L, embed_dim)

        Args:
            x: attention head outputs
        
        Returns: combined heads
        """
        B, num_heads, L, head_dim = x.shape
        x = x.transpose(1, 2)
        return x.contiguous().view(B, L, num_heads * head_dim)
    
    def forward(self,
                Q: Tensor,
                K: Tensor,
                V: Tensor,
                mask: Tensor | None = None,
                ) -> Tensor:
        """Compute Multi-Head Attention

        Args:
            Q, K, V: Query, Keys, Values tensors (explained in detail in SelfAttention.forward())
            mask: Attention mask of shape (B, h, L, L) where masked positions are 0 and unmasked are 1

        Returns:
            multi_head_attention: computed multi-head attention
        """
        # Project Q, K, V, and split outputs into heads
        Q = self._split_heads(self.Q_projection(Q))
        K = self._split_heads(self.K_projection(K))
        V = self._split_heads(self.V_projection(V))
        
        # Ensure mask shape compatiblity
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # Compute attention and combine heads
        multi_head_attention = self._combine_heads(self.self_attention(Q, K, V, mask))

        return multi_head_attention