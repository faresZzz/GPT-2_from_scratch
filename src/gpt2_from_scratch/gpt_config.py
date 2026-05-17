from enum import Enum

class GPTConfig(Enum):
    GPT2_SMALL = {"emb_dim": 768, "n_layers": 12, "n_heads": 12, "model_size":"124M"}
    GPT2_MEDIUM = {"emb_dim": 1024, "n_layers": 24, "n_heads": 16, "model_size":"355M"}
    GPT2_LARGE = {"emb_dim": 1280, "n_layers": 36, "n_heads": 20, "model_size":"774M"}
    GPT2_XL = {"emb_dim": 1600, "n_layers": 48, "n_heads": 25, "model_size":"1558M"}

    @staticmethod
    def get_config(model):
        BASE_CONFIG = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 1024,  # Context length
            "drop_rate": 0.0,        # Dropout rate
            "qkv_bias": True         # Query-key-value bias
        }
        
        if model in GPTConfig:
            BASE_CONFIG.update(model.value)
            return BASE_CONFIG
        else:
            raise ValueError("Invalid model selection")


if __name__ == '__main__':
    # Test the GPTConfig class
    choose_model = GPTConfig.GPT2_MEDIUM
    config = GPTConfig.get_config(choose_model)
    print(config)


