import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time 
from .gpt_config import GPTConfig
from .gpt_model import GPTModel
from . import dataloaders, metrics, plots, text_generator

def load_dataset(file_path):
    """
    Load text data from a file.
    This function reads the entire content of a text file and returns it as a string.
    Args:
        file_path (str): Path to the text file to be loaded.
    Returns:
        str: The content of the file as a string.
    Raises:
        FileNotFoundError: If the file at file_path does not exist.
        PermissionError: If the file cannot be accessed due to permission issues.
        UnicodeDecodeError: If the file contains text that cannot be decoded with UTF-8 encoding.
    """
    
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()
    return data

def create_dataloader(data, tokenizer, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):

    """
    Creates a PyTorch DataLoader for GPT-style pre-training on text data.
    This function processes raw text into a format suitable for training language models
    by tokenizing the text using the GPT-2 tokenizer and creating overlapping sequences
    of tokens with the specified length and stride.
    Parameters
    ----------
    txt : str
        Raw text data to be processed for pre-training.
    batch_size : int, optional
        Number of samples per batch (default: 4).
    max_length : int, optional
        Maximum length of each sequence in tokens (default: 256).
    stride : int, optional
        Number of tokens to stride between sequence windows (default: 128).
    shuffle : bool, optional
        Whether to shuffle the dataset (default: True).
    drop_last : bool, optional
        Whether to drop the last incomplete batch (default: True).
    num_workers : int, optional
        Number of worker processes for data loading (default: 0).
    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader that yields batches of tokenized sequences for training.
    """


    # Create dataset
    dataset = dataloaders.GPTDatasetPretrainingOverText(data, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

def train_model(model, train_loader, val_loader, optimizer, num_epochs, eval_freq, eval_iter, start_context, tokenizer):

    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()  
        
        # Iterate over the training data
        for input_batch, target_batch in train_loader:
            # Reset loss gradients from previous batch iteration
            optimizer.zero_grad() 

            # Compute the loss for the batch
            loss = metrics.calc_loss_batch(input_batch=input_batch, target_batch=target_batch, model=model)

            # Backpropagate the loss
            loss.backward()

            # Update model weights using loss gradients and optimizer
            optimizer.step() 

            # Track the number of tokens seen
            tokens_seen += input_batch.numel() 

            # Increment the global step
            global_step += 1

            # Optional evaluation step at the specified frequency
            if global_step % eval_freq == 0: 
                # Evaluate the model on the training and validation sets
                train_loss, val_loss = metrics.evaluate_model(model=model, train_loader=train_loader, val_loader=val_loader, num_batches=eval_iter, loss_fn=metrics.calc_loss_batch)

                # Append the losses and tokens seen
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                # Print the losses
                print(f"Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss: {val_loss:.3f} "
                      f"Perplexity loss {metrics.perplexity(train_loss):.3f}, Val loss {metrics.perplexity(val_loss):.3f} \n")

        # Print a sample text after each epoch
        print(text_generator.generate(text=start_context, model=model, tokenizer=tokenizer, max_length=50).replace("\n", " "))# Compact print format

    return train_losses, val_losses, track_tokens_seen

def train(model, train_loader, val_loader, optimizer, tokenizer,  num_epochs=10, eval_freq=5, eval_iter=0, start_context="Every effort moves you" ): 
    """
    Train a language model using provided data loaders.
    This function handles the training process of a language model, including evaluation at
    specified intervals. It tracks training and validation losses, as well as the total
    number of tokens processed during training.
    Parameters
    ----------
    model : torch.nn.Module
        The language model to be trained.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    optimizer : torch.optim.Optimizer
        The optimizer used for updating model parameters.
    device : torch.device
        The device (CPU or GPU) where the model and data will be moved.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for processing text data.
    num_epochs : int, optional
        Number of training epochs, by default 10.
    eval_freq : int, optional
        Frequency of evaluation in iterations, by default 5.
    eval_iter : int, optional
        Starting evaluation iteration, by default 0.
    start_context : str, optional
        Initial text for model evaluation, by default "Every effort moves you".
    Returns
    -------
    tuple
        A tuple containing:
        - train_losses (list): Training losses history.
        - val_losses (list): Validation losses history.
        - tokens_seen (int): Total number of tokens processed during training.
    Notes
    -----
    This function wraps the actual training implementation in `train_model` and handles
    timing and device management.
    """
    start_time = time.time()
    
    train_losses, val_losses, tokens_seen = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        tokenizer=tokenizer,
        num_epochs=num_epochs,
        eval_freq=eval_freq, 
        eval_iter=eval_iter,
        start_context=start_context
    )


    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    return train_losses, val_losses, tokens_seen

def main(): 
    """
    Main function for GPT-2 model pretraining.
    This function:
    - Configures and initializes a GPT-2 model (using GPT2_SMALL config)
    - Sets up the computing device (CUDA, MPS, or CPU)
    - Defines training hyperparameters
    - Loads the tokenizer and dataset
    - Performs sanity checks on dataset size
    - Creates data loaders for training and validation
    - Initializes the model and optimizer
    - Trains the model for the specified number of epochs
    - Plots training and validation losses
    The function uses the 'the-verdict.txt' dataset and the GPT-2 tokenizer.
    No parameters are required as all configurations are set within the function.
    """
    torch.manual_seed(123)

    # Choose the model configuration
    CHOOSE_MODEL = GPTConfig.GPT2_SMALL
    GPT_CONFIG_124M = GPTConfig.get_config(CHOOSE_MODEL)
    GPT_CONFIG_124M["context_length"] = 256
    GPT_CONFIG_124M["drop_rate"] = 0.2

    # Set the hyperparameters
    max_length = GPT_CONFIG_124M["context_length"]
    stride = GPT_CONFIG_124M["context_length"]
    batch_size = 2
    num_epochs = 5
    train_ratio = 0.90
    eval_freq = 5
    eval_iter = 5
    num_workers = 0
    learning_rate = 0.0004
    weight_decay = 0.1

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using {device} device.")

    

    print ("Hyperparameters set.")
    print(f"{max_length=}, {stride=}, {batch_size=}, {num_epochs=}, {train_ratio=}, {eval_freq=}, {eval_iter=}")
      

    # Load the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load the dataset
    data_dir = Path('data/raw')
    file_path = data_dir / "the-verdict.txt"
    dataset = load_dataset(file_path)

   
    # Calculate the total number of characters and tokens in the dataset
    total_characters = len(dataset)
    total_tokens = len(tokenizer.encode(dataset))
    print("Characters:", total_characters)
    print("Tokens:", total_tokens)


    # Sanity check
    if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the training loader. "
            "Try to lower the `GPT_CONFIG_124M['context_length']` or "
            "increase the `training_ratio`")

    if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the validation loader. "
            "Try to lower the `GPT_CONFIG_124M['context_length']` or "
            "decrease the `training_ratio`")
        

    # Split the dataset into training and validation sets
    split_idx = int(train_ratio * len(dataset))
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    train_loader = create_dataloader(
        data=train_data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=True,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = create_dataloader(
        data=val_data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers
    )

    print("Training and validation loaders created.")


    # load model 
    model = GPTModel(GPT_CONFIG_124M)
    # Move the model to the device
    model.to(device)
    print(f"Model moved on device {device}.")

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print("Model and optimizer initialized.")
    print("Starting training...")
    # Train the model
    train_losses, val_losses, tokens_seen = train(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        tokenizer=tokenizer, 
        num_epochs=num_epochs, 
        eval_freq=eval_freq, 
        eval_iter=eval_iter,
        start_context="Every effort moves you"
    )
    print("Training completed.")


    # Save the model and optimizer parameters
    model_path = Path(f"models/pretrained/pretraining{GPT_CONFIG_124M['model_size']}.pth")
    torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, 
        model_path
    )
    print(f"Model and optimizer parameters saved at {model_path}.")

    # Plot the losses
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plots.plot_train_val_losses(
        epochs_seen=epochs_tensor, 
        tokens_seen=tokens_seen, 
        train_losses=train_losses, 
        val_losses=val_losses, 
        save_path="reports/figures/pretraining_loss-plot.pdf",
        labels={
            "x": "Epochs", 
            "y": "Loss", 
            "train": "Training loss", 
            "validation": "Validation loss"}
        )

if __name__ == "__main__":
    main()
    exit(0)
