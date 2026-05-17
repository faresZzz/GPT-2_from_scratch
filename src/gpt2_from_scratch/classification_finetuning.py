import os
from  pathlib import Path
import re
import urllib.request
import zipfile
import ssl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time 
import tensorflow as tf
import tqdm
import numpy as np
import pandas as pd
import json
from functools import partial
import psutil
from . import dataloaders, downloader, metrics
from .gpt_config import GPTConfig
from .gpt_model import GPTModel



def preprocess_spam_data(train_ratio=0.7, validation_ratio=0.5):
    
    # Download and extract the spam data
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    working_path = Path("data/raw/sms_spam_collection")
    zip_name = "sms_spam_collection.zip"
    data_file_name = "SMSSpamCollection.tsv"
    data_file_path = working_path / data_file_name

    # Download and extract the data
    downloader.download_and_unzip_spam_data(url, zip_name, working_path, data_file_name)
    print("Data downloaded and extracted.")

    
    # Load the data into a DataFrame
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    
    # Preprocess the data
    df = create_balanced_dataset(df)
    # Map "ham" to 0 and "spam" to 1
    df["Label"] = df["Label"].map({"ham": 0, "spam": 1})
    print(f"HEAD DATA \n{df.head()} \n")
    print(f"COUNT DATA \n{df['Label'].value_counts()}\n")

    # Split the data into training, validation, and test sets
    train_df=df.sample(frac=train_ratio)
    test=df.drop(train_df.index)
    validation_df=test.sample(frac=validation_ratio/(1-train_ratio))
    test_df=validation_df.drop(validation_df.index)


    print(f"COUNT DATA \n{train_df['Label'].value_counts()}\n")
    print(f"COUNT DATA \n{validation_df['Label'].value_counts()}\n")
    print(f"COUNT DATA \n{test_df['Label'].value_counts()}\n")

    # Save the data to disk
    preproscessed_folder = Path("data/processed/sms_spam_collection")
    preproscessed_folder.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(preproscessed_folder / "train.csv", index=False)
    validation_df.to_csv(preproscessed_folder / "validation.csv", index=False)
    test_df.to_csv(preproscessed_folder / "test.csv", index=False)

    print(f"Data preprocessed and saved to disk at {preproscessed_folder}.")

    return train_df, validation_df, test_df

def create_balanced_dataset(df):
    
    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]
    
    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam)
    
    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df

def create_dataloader(df, tokenizer, batch_size, max_length=None, shuffle=True, num_workers=0, drop_last=True):

    # Create a SpamDataset
    dataset = dataloaders.SpamDataset(
    dataframe=df,
    max_length=max_length,
    tokenizer=tokenizer
    )

    # Create a DataLoader
    dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    drop_last=drop_last,
    )

    return dataloader

def train_classifier(model, train_loader, val_loader, optimizer, num_epochs, eval_freq, eval_iter):

    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:

            # Reset loss gradients from previous batch iteration
            optimizer.zero_grad() 

            # Calculate loss 
            loss = metrics.calc_loss_batch_last_logits(input_batch=input_batch, target_batch=target_batch, model=model)
            
            # Calculate loss gradients
            loss.backward() 

            # Update model weights using loss gradients
            optimizer.step() 

            # New: track examples instead of tokens seen
            examples_seen += input_batch.shape[0] 
            global_step += 1

            ## 130 batches: training, eval_Freq = 50 --> after 50 batches are processed in each epoch, we print train loss and val loss

            # Optional evaluation step
            if global_step % eval_freq == 0:
                # Evaluate the model on the training and validation sets
                train_loss, val_loss = metrics.evaluate_model(model=model, train_loader=train_loader, val_loader=val_loader, loss_fn=metrics.calc_loss_batch_last_logits)
                
                # Append the losses to track the training progress
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = metrics.calc_accuracy_loader(data_loader=train_loader, model=model, num_batches=eval_iter)
        val_accuracy = metrics.calc_accuracy_loader(data_loader=val_loader, model=model, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

def train(model, optimizer,  train_loader, val_loader, device, num_epochs, eval_freq, eval_iter):
    start_time = time.time()

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier(
        model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    
    return train_losses, val_losses, train_accs, val_accs, examples_seen

def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"reports/figures/{label}-plot.pdf")
    plt.show()

def test_model(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
    # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"

def main():
    # Seed the random number generator for reproducibility
    # torch.manual_seed(123)

    # Choose the model configuration
    CHOOSE_MODEL = GPTConfig.GPT2_SMALL
    GPT_CONFIG_124M = GPTConfig.get_config(CHOOSE_MODEL)
    GPT_CONFIG_124M["drop_rate"] = 0.2

    # Set the hyperparameters
    MAX_LENGTH = GPT_CONFIG_124M["context_length"]
    STRIDE = GPT_CONFIG_124M["context_length"]
    BATCH_SIZE = 8
    NUM_EPOCHS = 5
    TRAIN_RATIO = 0.7
    VALIDATION_RATIO = 0.1
    EVAL_FREQ = 50
    EVAL_ITER = 5
    NUM_WORKERS = 0
    NUM_CLASSES = 2

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using {device} device.")


    # Preprocess the spam data
    train_df, validation_df, test_df = preprocess_spam_data(TRAIN_RATIO, VALIDATION_RATIO)

    #  Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load the data into PyTorch Dataset and DataLoader
    train_dataloader = create_dataloader(train_df, tokenizer, batch_size=BATCH_SIZE, max_length=None)
    validation_dataloader = create_dataloader(validation_df, tokenizer, batch_size=BATCH_SIZE, max_length=train_dataloader.dataset.max_length)
    test_dataloader = create_dataloader(test_df, tokenizer, batch_size=BATCH_SIZE, max_length=train_dataloader.dataset.max_length)

    print(f"{len(train_dataloader)} training batches")
    print(f"{len(validation_dataloader)} validation batches")
    print(f"{len(test_dataloader)} test batches")

    print("Data loaded into PyTorch DataLoaders.")

    # Sanity check
    assert train_dataloader.dataset.max_length <= GPT_CONFIG_124M["context_length"], (
    f"Dataset length {train_dataloader.dataset.max_length} exceeds model's context "
    f"length {GPT_CONFIG_124M['context_length']}. Reinitialize data sets with "
    f"`max_length={GPT_CONFIG_124M['context_length']}`"
    )


    # Initialize the GPT model
    model = GPTModel(GPT_CONFIG_124M)
    settings, params = downloader.download_and_load_gpt2(model_size=GPT_CONFIG_124M["model_size"], models_dir="models/gpt2")
    downloader.load_weights_into_gpt(model, params)

    # Block the model's parameters from being updated
    for param in model.parameters():
        param.requires_grad = False

    # Add a classification head to the model --> grad=True
    model.out_head = torch.nn.Linear(in_features=GPT_CONFIG_124M["emb_dim"], out_features=num_classes)
    
    #set learnable parameters of last transformer block 
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    #set learnable parameters of last norm layer
    for param in model.final_norm.parameters():
        param.requires_grad = True

    # Move the model to the device
    model.to(device)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=2e-5)

    print("Model and optimizer initialized.")
    print("Starting training...")

    # Train the model
    train_losses, val_losses, train_accs, val_accs, examples_seen = train(
    model=model, 
    train_loader=train_dataloader, 
    val_loader=validation_dataloader, 
    optimizer=optimizer, 
    device=device,
    num_epochs=num_epochs, 
    eval_freq=eval_freq, 
    eval_iter=eval_iter,
    )

    # Save the model
    model_path = Path(f"models/finetuning/spam_classifier{GPT_CONFIG_124M['model_size']}.pth")
    torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, 
        model_path
    )
    print(f"Model and optimizer parameters saved at {model_path}.")
    print("Finetuning completed.")

    # Plot the training and validation losses
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses, label=f"finetune_{num_epochs}_loss_lr{3e-5}_wd{2e-5}")

    # Plot the training and validation accuracies
    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label=f"finetune_{num_epochs}_accuracy_lr{3e-5}_wd{2e-5}")

  
    # load the model
    # model_path = Path("models/finetuning/spam_classifier.pth")
    # model = GPTModel(GPT_CONFIG_124M)
    # # Add a classification head to the model --> grad=True
    # model.out_head = torch.nn.Linear(in_features=GPT_CONFIG_124M["emb_dim"], out_features=num_classes)
    # # Load the model state dict
    # model.load_state_dict(torch.load(model_path)["model_state_dict"])
    # print(f"Model loaded from {model_path}.")
    # # Block the model's parameters from being updated
    # for param in model.parameters():
    #     param.requires_grad = False
    # Move the model to the device
    # model.to(device)

    # compute the final accuracy
    train_accuracy = metrics.calc_accuracy_loader(train_dataloader, model, device)
    val_accuracy = metrics.calc_accuracy_loader(validation_dataloader, model, device)
    test_accuracy = metrics.calc_accuracy_loader(test_dataloader, model, device)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}% \n\n")


    # Test the classifier
    text = "WINNER! You have won a free vacation. Claim now!"
    classified_label = test_model(text, model, tokenizer, device, max_length=max_length)
    print(f"Text: {text}")
    print(f"Classified as: {classified_label}\n\n")

    text = "Bonjour comment vas-tu?"
    classified_label = test_model(text, model, tokenizer, device, max_length=max_length)
    print(f"Text: {text}")
    print(f"Classified as: {classified_label}\n\n")

    text = "You have a new message. Click here to read it."
    classified_label = test_model(text, model, tokenizer, device, max_length=max_length)
    print(f"Text: {text}")
    print(f"Classified as: {classified_label} \n\n")




if __name__ == "__main__":  
   main()
   exit(0)
