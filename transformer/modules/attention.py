"""
From 'Attention is All You Need' Paper

Contains implementations for three types of Attention:
    1) Self-Attention
    2) Masked Self-Attention
    3) Encoder-Decoder Attention
"""
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod

class Attention(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def _scaled_dot_product_attention(self,
                                      Q: Tensor,
                                      K: Tensor,
                                      V: Tensor,
                                      mask: Tensor | None = None,) -> Tensor:
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
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass