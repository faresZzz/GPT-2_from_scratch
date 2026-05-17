# Description: Transformer block implementation
import torch
import torch.nn as nn
from . import attention


class LayerNorm(nn.Module):
    """
    Layer Normalization module for neural networks.
    This class implements Layer Normalization as described in the paper:
    "Layer Normalization" by Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
    (https://arxiv.org/abs/1607.06450)
    Layer Normalization normalizes the inputs across features, rather than across batch 
    dimensions. It learns an affine transformation (scale and shift) which is applied 
    after the normalization.
    Args:
        emb_dim (int): The size of the embedding dimension to normalize.
    Attributes:
        eps (float): A small constant added to the variance for numerical stability.
        scale (nn.Parameter): Learnable scaling parameter of shape (emb_dim).
        shift (nn.Parameter): Learnable shifting parameter of shape (emb_dim).
    Example:
        >>> layer_norm = LayerNorm(768)
        >>> x = torch.randn(32, 10, 768)  # [batch_size, seq_len, emb_dim]
        >>> output = layer_norm(x)  # Has the same shape as input
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5

        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # Compute mean and variance
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # Normalize the input
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    """Gaussian Error Linear Unit (GELU) activation function.
    This implements the GELU activation function as described in the paper:
    "Gaussian Error Linear Units (GELUs)" by Dan Hendrycks and Kevin Gimpel.
    The GELU activation is defined as:
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    This is a smooth approximation to the ReLU function and is used in models 
    like BERT, GPT-2, and other Transformer architectures.
    Args:
        x (torch.Tensor): Input tensor
    Returns:
        torch.Tensor: Output after applying GELU activation
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    """Feed-forward neural network module for transformer architectures.
    This module implements the standard feed-forward network used in transformer models,
    consisting of two linear transformations with a GELU activation in between,
    as described in "Attention Is All You Need" (Vaswani et al., 2017) with GELU
    instead of the original ReLU.
    The first linear layer expands the input dimension by a factor of 4,
    followed by a GELU activation, and the second layer projects back to the original dimension.
    Args:
        cfg (dict): Configuration dictionary containing:
            - emb_dim (int): Embedding dimension of the model.
    Methods:
        forward(x): Apply the feed-forward network to the input tensor.
            Args:
                x (torch.Tensor): Input tensor of shape (..., emb_dim)
            Returns:
                torch.Tensor: Output tensor of shape (..., emb_dim)
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), ## Expansion
            GELU(), ## Activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), ## Contraction
        )

    def forward(self, x):
        return self.layers(x)
    
class TransformerBlock(nn.Module):
    """
    TransformerBlock implements a standard transformer encoder block with pre-normalization.
    This block applies the following operations in sequence:
    1. Layer normalization (pre-norm architecture)
    2. Multi-head self-attention
    3. Residual connection with dropout
    4. Layer normalization
    5. Feed-forward network
    6. Residual connection with dropout
    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing:
        - emb_dim (int): Embedding dimension size
        - context_length (int): Maximum sequence length
        - n_heads (int): Number of attention heads
        - drop_rate (float): Dropout probability
        - qkv_bias (bool): Whether to include bias in query, key, value projections
    Returns
    -------
    torch.Tensor
        Output tensor with same shape as input: [batch_size, num_tokens, emb_dim]
    Notes
    -----
    This implementation follows the pre-norm variant where normalization is
    applied before the attention and feed-forward blocks, which can improve
    training stability.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.att = attention.MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        # 2*4*768
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
        # 2*4*768


if __name__ == '__main__':
    torch.manual_seed(123)
    smoke_config = {
        "emb_dim": 16,
        "context_length": 8,
        "n_heads": 4,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    x = torch.rand(2, 4, 16)
    block = TransformerBlock(smoke_config)
    output = block(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    
