# Description: GPT model implementation
import torch
import torch.nn as nn
from .transformer import LayerNorm, TransformerBlock

class GPTModel(nn.Module):
    """
    A PyTorch implementation of the GPT (Generative Pre-trained Transformer) model architecture.
    This model follows the transformer decoder architecture used in GPT models, with
    token embeddings, positional embeddings, multiple transformer blocks, and an output head
    for next-token prediction.
    This block applies the following operations in sequence:
    1. Token embedding
    2. Positional embedding
    3. Dropout
    4. Transformer blocks
    5. Layer normalization
    6. Output head
    Args:
        cfg (dict): Configuration dictionary containing model parameters:
            - vocab_size (int): Size of the vocabulary
            - emb_dim (int): Dimension of the embeddings
            - context_length (int): Maximum sequence length
            - drop_rate (float): Dropout probability
            - n_layers (int): Number of transformer blocks
    Attributes:
        tok_emb (nn.Embedding): Token embedding layer
        pos_emb (nn.Embedding): Position embedding layer
        drop_emb (nn.Dropout): Dropout layer for embeddings
        trf_blocks (nn.Sequential): Stack of transformer blocks
        final_norm (LayerNorm): Layer normalization before output
        out_head (nn.Linear): Output projection to vocabulary logits
    Returns:
        torch.Tensor: Logits for next-token prediction with shape [batch_size, seq_len, vocab_size]
    """


    def __init__(self, cfg):
        super().__init__()
        # Token and positional embeddings layers dimensions are the same as the embedding dimension
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) 
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # Dropout layer
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        # Layer normalization
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # Output head: dimension of the output is the same as the vocabulary size
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        # get batch size and sequence length
        batch_size, seq_len = in_idx.shape

        # Token and positional embeddings
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        
        # Apply dropout to the embeddings
        x = self.drop_emb(x)

        # Transformer blocks
        x = self.trf_blocks(x)

        # Apply layer normalization
        x = self.final_norm(x)

        # Output head
        logits = self.out_head(x)

        return logits
    

if __name__ == '__main__':
    torch.manual_seed(123)
    smoke_config = {
        "vocab_size": 64,
        "context_length": 8,
        "emb_dim": 16,
        "n_layers": 2,
        "n_heads": 4,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    batch = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    model = GPTModel(smoke_config)
    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")


 
