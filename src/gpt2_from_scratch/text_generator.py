import torch

def text_to_token_ids(text, tokenizer, allowed_special={'<|endoftext|>'}):
    """
    Convert input text to token IDs using the provided tokenizer.
    This function encodes the input text into token IDs and converts the result
    into a PyTorch tensor with an added batch dimension.
    Args:
        text (str): The input text to tokenize and convert to IDs.
        tokenizer: The tokenizer to use for encoding the text. Should have an
                   'encode' method that supports the 'allowed_special' parameter.
    Returns:
        torch.Tensor: A tensor of shape (1, sequence_length) containing the token IDs,
                     where the first dimension is the batch dimension.
    """
    # Encode the text and convert to tensor
    encoded = tokenizer.encode(text, allowed_special=allowed_special)
    
    # add batch dimension
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) 
    
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    # remove batch dimension
    flat = token_ids.squeeze(0) 
    
    # Decode the token IDs back to text
    decoded = tokenizer.decode(flat.tolist())

    return decoded

def _generate(model, idx, max_length, context_size=None, temperature=0.0, top_k=None, eos_id=None):
    """
    Generate text by autoregressively sampling from a language model.
    This function takes a pre-trained language model and an initial sequence of token indices,
    then generates additional tokens one by one up to a specified maximum. It supports various
    generation strategies including deterministic (greedy) sampling, temperature-controlled
    sampling, and top-k filtering.
    Parameters
    ----------
    model : torch.nn.Module
        The language model that takes token indices and outputs logits
    idx : torch.Tensor
        Initial token indices tensor of shape (batch_size, seq_len)
    max_length : int
        Maximum number of new tokens to generate
    context_size : int
        Number of previous tokens to consider as context for each prediction
    temperature : float, default=0.0
        Controls randomness in generation:
        - If 0.0, uses greedy sampling (argmax)
        - If >0.0, uses temperature-scaled sampling with higher values increasing randomness
    top_k : int or None, default=None
        If provided, restricts sampling to the top k most probable tokens
    eos_id : int or None, default=None
        If provided, stops generation when this token ID is encountered
    Returns
    -------
    torch.Tensor
        Extended token indices tensor of shape (batch_size, seq_len + generated_len)
        where generated_len <= max_length
    """
    # Set the model to evaluation mode
    training_mode = model.training
    model.eval()

    # Get model device
    device = next(model.parameters()).device

    # Move input to the same device as the model
    idx = idx.to(device)

    # context_size
    if context_size is None:
        context_size = model.pos_emb.weight.shape[0]

    # Get logits, and only focus on last time step
    for _ in range(max_length):
        # Crop current context if it exceeds the supported context size
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # Stop generating early if end-of-sequence token is encountered and eos_id is specified
        if idx_next == eos_id:
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    # Restore the model to its original training mode
    model.train(training_mode)
    return idx

def generate(text, model, tokenizer, max_length, context_size=None, temperature=0.0, top_k=None, eos_id=None):
    """
    Generate text using a language model based on an initial text prompt.
    This function takes a text prompt, encodes it using the provided tokenizer,
    generates new tokens using the provided model, and then decodes the tokens
    back to text.
    Args:
        text (str): The input text prompt to start generation from.
        model: The language model used for text generation.
        tokenizer: The tokenizer used to convert between text and token IDs.
        max_length (int): Maximum number of new tokens to generate.
        context_size (int): Maximum context size the model can process.
        temperature (float, optional): Controls randomness in generation.
            Values closer to 0 make the generation more deterministic.
            Default is 0.0 (deterministic).
        top_k (int, optional): If specified, limits sampling to the top k most
            likely tokens at each step. Default is None.
        eos_id (int, optional): Token ID that signals the end of sequence.
            Generation stops if this token is produced. Default is None.
    Returns:
        str: The generated text including the original prompt.
    """
    # context_size
    if context_size is None:
        context_size = model.pos_emb.weight.shape[0]

    # Encode the text prompt
    idx = text_to_token_ids(text, tokenizer).to(next(model.parameters()).device)

    # Generate new tokens
    out = _generate(
        model=model, 
        idx=idx, 
        max_length=max_length, 
        context_size=context_size, 
        temperature=temperature, 
        top_k=top_k, 
        eos_id=eos_id)

    # Decode the generated tokens
    text_gen = token_ids_to_text(out, tokenizer)

    return text_gen

if __name__ == "__main__":
    from gpt_config import GPTConfig
    from gpt_model import GPTModel
    import tiktoken

    # Choose the model configuration
    CHOOSE_MODEL = GPTConfig.GPT2_SMALL
    GPT_CONFIG_124M = GPTConfig.get_config(CHOOSE_MODEL)

    # Load the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(GPT_CONFIG_124M)

    text = "The quick brown fox jumps over the lazy dog."

    # Converte text to token ids
    token_ids = text_to_token_ids(text, tokenizer)
    print(f"\nText: {text}")
    print(f"Token IDs: {token_ids}")

    # Convert token ids back to text
    text_decoded = token_ids_to_text(token_ids, tokenizer)
    print(f"\nToken IDs: {token_ids}")
    print(f"Text: {text_decoded}")

    # Generate next token selecting argmax
    response = generate(text, model, tokenizer, 15, GPT_CONFIG_124M["context_length"], temperature=0.0, top_k=None)
    print(f"\nPrompt: {text}")
    print(f"Response (argmax): {response}")

    # Generate a text sample with temperature scaling
    response = generate(text, model, tokenizer, 15, GPT_CONFIG_124M["context_length"], temperature=1.4, top_k=None)
    print(f"\nPrompt: {text}")
    print(f"Response (temperature=1.4): {response}")

    # Generate a text sample with top-k sampling and temperature scaling
    response = generate(text, model, tokenizer, 15, GPT_CONFIG_124M["context_length"], top_k=25, temperature=1.4)
    print(f"\nPrompt: {text}")
    print(f"Response (top_k=25, temperature=1.4): {response}")

