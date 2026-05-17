# Description: Implement the CausalAttention, MultiHeadAttentionWrapper, and MultiHeadAttention classes.
import torch
import torch.nn as nn
import numpy as np


class CausalAttention(nn.Module):
    """
    Causal self-attention module that implements the masked multi-head attention mechanism
    used in autoregressive transformer models like GPT-2.
    This attention module ensures each position can only attend to previous positions and itself,
    maintaining the autoregressive property by using an upper triangular mask. It prevents information
    leakage from future tokens during training and inference.
    Args:
        d_in (int): Input dimension of the embeddings.
        d_out (int): Output dimension for query, key, value projections.
        context_length (int): Maximum sequence length supported by the model.
        dropout (float): Dropout probability applied to attention weights.
        qkv_bias (bool, optional): Whether to include bias terms in query, key, value projections. 
                                  Defaults to False.
    Input:
        x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, d_in)
    Returns:
        torch.Tensor: Context vectors of shape (batch_size, num_tokens, d_out)
    """

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        # Linear layers for queries, keys, and values
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # Dropout matrix using the dropout rate
        self.dropout = nn.Dropout(dropout) # New
        # Upper triangular mask shifted by 1 to the right saved as buffer (not learnable)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # compute attention scores
        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
        
        # apply TRIU mask to attention scores and set to -inf
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        
        #compute attention weights with softmax
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # apply dropout to attention weights
        attn_weights = self.dropout(attn_weights) # New

        # compute context vectors
        context_vec = attn_weights @ values

        return context_vec
    

class MultiHeadAttentionWrapper(nn.Module):
    """Multi-Head Causal Attention wrapper.
    This module implements multi-head attention by creating multiple parallel
    causal attention heads and concatenating their outputs. Each head projects
    input data into its own query, key, and value spaces, and computes attention
    independently.
    Args:
        d_in (int): Input embedding dimension
        d_out (int): Output dimension per head
        context_length (int): Maximum sequence length for causal masking
        dropout (float): Dropout probability for attention weights
        num_heads (int): Number of attention heads
        qkv_bias (bool, optional): Whether to include bias terms in query, key, and value
            projections. Defaults to False.
    Shape:
        - Input: (batch_size, seq_len, d_in)
        - Output: (batch_size, seq_len, d_out * num_heads)
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # Create multiple heads
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module as described in "Attention Is All You Need" (Vaswani et al., 2017).
    This implementation includes causal masking for use in autoregressive models, making it suitable
    for decoder-only transformer architectures like GPT-2. The attention mechanism splits the input
    into multiple heads to allow the model to jointly attend to information from different
    representation subspaces at different positions.
    Parameters
    ----------
    d_in : int
        Input dimension (embedding dimension).
    d_out : int
        Output dimension after attention. Must be divisible by num_heads.
    context_length : int
        The maximum sequence length this attention layer will handle.
    dropout : float
        Dropout probability applied to attention weights.
    num_heads : int
        Number of attention heads. d_out must be divisible by this value.
    qkv_bias : bool, default=False
        Whether to include bias terms in the query, key, and value projections.
    Attributes
    ----------
    head_dim : int
        Dimension of each attention head (d_out / num_heads).
    mask : torch.Tensor
        Causal mask tensor of shape (context_length, context_length) where upper triangular
        elements are set to 1 and will be converted to -inf in the attention scores.
    Methods
    -------
    forward(x)
        Computes multi-head self-attention over the input tensor.
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_tokens, d_in)
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_tokens, d_out) containing the 
            attention-weighted information from the sequence.
    """
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # Check if d_out is divisible by num_heads
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        # Linear layers for queries, keys, and values
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Linear layer for the output projection
        self.out_proj = nn.Linear(d_out, d_out)  

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Upper triangular mask shifted by 1 to the right
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # Get the batch size, number of tokens, and input dimension
        b, num_tokens, d_in = x.shape

        # Project input tokens to queries, keys, and values
        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # compute the softmax of the attention scores first normalizing by the sqrt of the head_dim
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape:  transpose (b, num_heads, nums_tokens, head_dim) => (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads (Flatten ), where self.d_out = self.num_heads * self.head_dim
        # Shape: (b, num_tokens, num_heads, head_dim) -> (b, num_tokens, d_out)
        #contiguous() is used to ensure that the tensor is stored in a contiguous chunk of memory
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec
    

if __name__ == '__main__':
   
   # Setup tests
    torch.manual_seed(123)
    d_in = 3
    d_out = 2
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your     (x^1)
        [0.55, 0.87, 0.66], # journey  (x^2)
        [0.57, 0.85, 0.64], # starts   (x^3)
        [0.22, 0.58, 0.33], # with     (x^4)
        [0.77, 0.25, 0.10], # one      (x^5)
        [0.05, 0.80, 0.55]] # step     (x^6)
        )
    batch = torch.stack((inputs, inputs), dim=0)
    context_length = batch.shape[1]


     # Test the CausalAttention module
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch)
    print(f"context_vecs.shape: {context_vecs.shape}\n\n")  # Expected output: torch.Size([2, 6, 2])


    # Test the MultiHeadAttentionWrapper module
    mhaw = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mhaw(batch)
    print(context_vecs)
    print(f"context_vecs.shape: {context_vecs.shape}\n\n") # Expected output: torch.Size([2, 6, 4])


    # Test the MultiHeadAttention module
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print(f"context_vecs.shape: {context_vecs.shape}\n\n") # Expected output: torch.Size([2, 6, 4])





