# codec/nn/transformer.py

import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer
    
    Args:
        d_model: Dimension of the model
        max_len: Maximum sequence length
    """
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, T, D]
        Returns:
            Tensor of shape [B, T, D] with positional encoding added
        """
        # x: [B, T, D]
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)