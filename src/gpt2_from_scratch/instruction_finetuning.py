import os
from  pathlib import Path
import urllib.request
import ssl
import torch
from torch.utils.data import DataLoader
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time 
import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import json
from functools import partial
from . import dataloaders, downloader, metrics, text_generator
from .gpt_config import GPTConfig
from .gpt_model import GPTModel



def preprocess_data(train_ratio=0.80, test_to_val_ration=0.50):
    # Download the instruction data
    folder_path = Path("data/raw/finetune_instruction")
    file_path = folder_path / "instruction-data.json"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"

    data = download_and_load_file(folder_path, file_path, url)
    print("Number of entries:", len(data))

    # Split the data into training, validation, and test sets
    train_data, temp_data = train_test_split(data, test_size=float(1 - train_ratio))
    val_data, test_data = train_test_split(temp_data, test_size=float(1 - test_to_val_ration))

    labels = ["instruction", "input", "output"]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))

    # convert the data to a pandas DataFrame
    train_df = pd.DataFrame(train_data, columns=labels)
    val_df = pd.DataFrame(val_data, columns=labels)
    test_df = pd.DataFrame(test_data, columns=labels)

    # Save the data to disk
    folder_path = Path("data/processed/finetune_instruction")
    train_file_path = folder_path / "train.json"
    val_file_path = folder_path / "val.json"
    test_file_path = folder_path / "test.json"
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    train_df.to_json(train_file_path, orient="records", lines=True)
    val_df.to_json(val_file_path, orient="records", lines=True)
    test_df.to_json(test_file_path, orient="records", lines=True)

    print(f"Data preprocessed and saved to disk at {folder_path}.")

    return train_data, val_data, test_data

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

def download_and_load_file(folder_path, file_path, url):

    # Create a secure SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Download the file if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url, context=ssl_context) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "wb", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data
 
def custom_collate(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    # Custom collate function to pad sequences and prepare inputs and targets
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))

        # Create input target pairs
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

def create_dataloaders(data, tokenizer, batch_size, max_length=None, shuffle=True, num_workers=0, drop_last=True, device="cpu"):
    
   
    # Partially apply the custom_collate function with the specified parameters
    partial_collate = partial(custom_collate, device=device, allowed_max_length=max_length)

    # Create a PyTorch Dataset and DataLoader
    dataset = dataloaders.InstructionDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=partial_collate, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

    return dataloader

def train_model(model, tokenizer, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context):
    """
    Train a neural language model with periodic evaluation and text generation.
    This function implements a simple training loop for language models, including
    regular evaluation on validation data and example text generation after each
    epoch to visualize the model's progress.
    Parameters:
    ----------
    model : torch.nn.Module
        The language model to be trained.
    tokenizer : object
        Tokenizer with methods to convert between text and token IDs.
    train_loader : torch.utils.data.DataLoader
        DataLoader providing batches of training data.
    val_loader : torch.utils.data.DataLoader
        DataLoader providing batches of validation data.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model weights.
    device : torch.device
        Device to run the model on (CPU or GPU).
    num_epochs : int
        Number of complete passes through the training dataset.
    eval_freq : int
        Number of steps between evaluation points.
    eval_iter : int
        Number of batches to use during evaluation.
    start_context : str
        Initial text prompt for generation examples.
    Returns:
    -------
    tuple
        A tuple containing:
        - train_losses (list): Training losses at each evaluation point
        - val_losses (list): Validation losses at each evaluation point
        - track_tokens_seen (list): Cumulative tokens processed at each evaluation
    """

    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train() 
        
        for input_batch, target_batch in train_loader:
            # Reset loss gradients from previous batch iteration
            optimizer.zero_grad() 

            # Compute loss function for the batch
            loss = metrics.calc_loss_batch(input_batch, target_batch, model, device)

            # Calculate loss gradients
            loss.backward() 

            # Update model weights using loss gradients
            optimizer.step()

            # Track the number of tokens seen
            tokens_seen += input_batch.numel()

            # Increment the global step counter
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                # Evaluate the model on training and validation data 
                train_loss, val_loss = metrics.evaluate_model(model, train_loader, val_loader, device, eval_iter, loss_func=metrics.calc_loss_batch)

                # Append the losses and tokens seen to the tracking lists
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")


        # Print a sample text after each epoch

        out_text = text_generator.generate(text=start_context, model=model, tokenizer=tokenizer, max_length=50, context_size=256, eos_id=50256)
        print(out_text.replace("\n", " ")) # Print the generated text compact format 
    
    return train_losses, val_losses, track_tokens_seen

def train(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):

    start_time = time.time()

    train_losses, val_losses, tokens_seen = train_model( model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, device=device, num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter, start_context=start_context, tokenizer=tokenizer)

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    return train_losses, val_losses, tokens_seen

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, file_path):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(file_path)
    plt.show()

def test_model(model,  tokenizer, text=None, special_prompt=None, max_length=256, context_size=256, eos_id=50256):

    # Set the model to evaluation mode
    model.eval()

    # Set the input text
    if special_prompt is not None:
        input_text = special_prompt
    else: 
        # Format the input text 
        if text is None:
            raise ValueError("Input text is required")
        
        input_text = format_input(text)

    # Generate the output text
    model_response = text_generator.generate(text=input_text, model=model, tokenizer=tokenizer, max_length=max_length, context_size=context_size, eos_id=eos_id)

    model_response = model_response[len(input_text):].replace("### Response:", "").strip()

    return model_response

def ollamaEvaluation(json_data, json_key, Ollama_model):
    import psutil
    def _check_if_running(process_name):
        running = False
        for proc in psutil.process_iter(["name"]):
            if process_name in proc.info["name"]:
                running = True
                break
        return running

    if not _check_if_running("ollama"):
        raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
    print("Ollama running:")    

    scores = []
    for entry in tqdm.tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = downloader.query_model(prompt=prompt, model=Ollama_model, url="http://localhost:11434/api/chat")
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores

def main():
    # Seed the random number generator for reproducibility
    torch.manual_seed(123)

    # Choose the model configuration
    CHOOSE_MODEL = GPTConfig.GPT2_MEDIUM
    GPT_CONFIG_355M = GPTConfig.get_config(CHOOSE_MODEL)
    # GPT_CONFIG_355M["drop_rate"] = 0.1


    # Set the hyperparameters
    max_length = GPT_CONFIG_355M["context_length"]
    stride = GPT_CONFIG_355M["context_length"]

    batch_size = 8
    num_epochs = 5
    train_ratio = 0.70
    test_to_val_ration = 0.50
    eval_freq = 5
    eval_iter = 5
    num_workers = 0

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Move the model to the device
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")

    print(f"Using {device} device.")

    # Preprocess the spam data
    train_data, validation_data, test_data = preprocess_data(train_ratio=train_ratio, test_to_val_ration=test_to_val_ration)

    # Create the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load the data into PyTorch Dataset and DataLoader
    train_dataloader = create_dataloaders(data=train_data, tokenizer=tokenizer, batch_size=batch_size, max_length=max_length, num_workers=num_workers, device=device, shuffle=True, drop_last=True)
    validation_dataloader = create_dataloaders(data=validation_data, tokenizer=tokenizer, batch_size=batch_size, max_length=max_length, num_workers=num_workers, device=device, shuffle=False, drop_last=False)
    test_dataloader = create_dataloaders(data=test_data, tokenizer=tokenizer, batch_size=batch_size, max_length=max_length, num_workers=num_workers, device=device, shuffle=False, drop_last=False)

    print(f"{len(train_dataloader)} training batches")
    print(f"{len(validation_dataloader)} validation batches")
    print(f"{len(test_dataloader)} test batches")

    print("Data loaded into PyTorch DataLoaders.")

    # Instantiate the GPT model
    model = GPTModel(GPT_CONFIG_355M)
    _ , params = downloader.download_and_load_gpt2(model_size=GPT_CONFIG_355M["model_size"], models_dir="models/gpt2")
    downloader.load_weights_into_gpt(model, params)
    model.to(device)

    # Set the optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    print("Model and optimizer initialized.")
    print("Starting training...")

    # Train the model
    train_losses, val_losses, tokens_seen = train(model=model, train_loader=train_dataloader, val_loader=validation_dataloader, optimizer=optimizer, device=device, num_epochs=num_epochs,eval_freq=eval_freq, eval_iter=eval_iter, start_context=format_input(validation_data[0]), tokenizer=tokenizer)

    # Save the model
    model_path = Path(f"models/finetuning/instruction{GPT_CONFIG_355M['model_size']}.pth")
    torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, 
        model_path
    )
    print(f"Model and optimizer parameters saved at {model_path}.")
    print("Finetuning completed.")


    # load the model
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    print(f"Model loaded from {model_path}.")

    model.to(device)

    # Plot the training and validation losses
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, file_path="reports/figures/instruction_finetune_loss_plot.pdf")   

    # Test the model on exemple data

    input_text = test_data[0]
    model_response = test_model(text=input_text, model=model, tokenizer=tokenizer)

    print(f""" 
    ### Instruction:    
        {input_text["instruction"]}

    ### Input:
        {input_text["input"]}

    ### Expected output:
        {input_text["output"]}

    ### Model response:
        {model_response}

    """)

    # Evaluate the model on the test set
    print("Starting model testing...")
    for i, entry in tqdm.tqdm(enumerate(test_data), total=len(test_data)):
        model_response = test_model(text=entry, model=model, tokenizer=tokenizer)
        test_data[i]["model_response"] = model_response

    # Save the test data with model responses
    test_file_path = Path("data/processed/finetune_instruction/test_with_responses.json")
    with open(test_file_path, "w", encoding="utf-8") as file:
        json.dump(test_data, file, indent=4)

    print("Model testing completed.")


    # Evaluate the model using Ollama 
    #load the test data
    # test_data = None
    # with open(test_file_path, "r", encoding="utf-8") as file:
    #     test_data = json.load(file)
    
    # Check if Ollama is running
    ollama = "llama3.2:latest"
    scores = ollamaEvaluation(json_data=test_data, json_key="model_response", Ollama_model=ollama)
    print("Mean score:", np.mean(scores))
    print("Ollama evaluation completed.")


if __name__ == '__main__':
    main()
    exit(0)
