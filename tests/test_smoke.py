import torch
import importlib

from gpt2_from_scratch.attention import CausalAttention
from gpt2_from_scratch.gpt_model import GPTModel
from gpt2_from_scratch.text_generator import _generate, text_to_token_ids, token_ids_to_text


class TinyTokenizer:
    def __init__(self):
        self.vocab = {"hello": 1, "world": 2}
        self.inverse_vocab = {value: key for key, value in self.vocab.items()}

    def encode(self, text, allowed_special=None):
        return [self.vocab[token] for token in text.split()]

    def decode(self, ids):
        return " ".join(self.inverse_vocab[token_id] for token_id in ids)


class TinyNextTokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_emb = torch.nn.Embedding(8, 4)
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, idx):
        batch_size, seq_len = idx.shape
        logits = torch.zeros(batch_size, seq_len, 5, device=idx.device)
        logits[:, -1, 2] = 1.0
        return logits + self.anchor


def test_gpt_model_forward_shape_with_tiny_config():
    cfg = {
        "vocab_size": 32,
        "context_length": 8,
        "emb_dim": 16,
        "n_layers": 2,
        "n_heads": 4,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    model = GPTModel(cfg)
    tokens = torch.randint(0, cfg["vocab_size"], (2, 5))

    logits = model(tokens)

    assert logits.shape == (2, 5, cfg["vocab_size"])


def test_causal_attention_does_not_use_future_tokens_for_prefix_outputs():
    torch.manual_seed(123)
    attention = CausalAttention(
        d_in=4,
        d_out=4,
        context_length=6,
        dropout=0.0,
        qkv_bias=False,
    )
    x = torch.randn(1, 6, 4)
    changed_future = x.clone()
    changed_future[:, 3:, :] = torch.randn(1, 3, 4) * 10

    original = attention(x)
    modified = attention(changed_future)

    torch.testing.assert_close(original[:, :3, :], modified[:, :3, :])


def test_generation_utilities_with_tiny_tokenizer_and_model():
    tokenizer = TinyTokenizer()
    model = TinyNextTokenModel()

    ids = text_to_token_ids("hello world", tokenizer)
    generated = _generate(model=model, idx=ids, max_length=2, temperature=0.0)
    decoded = token_ids_to_text(ids, tokenizer)

    assert ids.tolist() == [[1, 2]]
    assert generated.tolist() == [[1, 2, 2, 2]]
    assert decoded == "hello world"


def test_package_modules_import_without_path_hacks():
    modules = [
        "attention",
        "classification_finetuning",
        "dataloaders",
        "downloader",
        "gpt_config",
        "gpt_model",
        "instruction_finetuning",
        "metrics",
        "plots",
        "pretraining",
        "text_generator",
        "tokenizer",
        "train",
        "transformer",
    ]

    for module in modules:
        assert importlib.import_module(f"gpt2_from_scratch.{module}")
