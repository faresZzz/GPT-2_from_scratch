import torch
import time 

from . import metrics, text_generator


def train_model_simple(model, tokenizer, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = metrics.calc_loss_batch(input_batch, target_batch, model)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel() # Returns the total number of elements (or tokens) in the input_batch.
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0: 
                train_loss, val_loss = metrics.evaluate_model(
                    model, train_loader, val_loader, loss_fn=metrics.calc_loss_batch, num_batches=eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")


        # Print a sample text after each epoch
        encoded = text_generator.text_to_token_ids(start_context, tokenizer).to(device)
        out_ids = text_generator._generate(
            idx=encoded,
            model=model,
            max_length=50

        )
        out_text = text_generator.token_ids_to_text(out_ids, tokenizer)
        print(out_text.replace("\n", " ")) # Print the generated text compact format 
    
    return train_losses, val_losses, track_tokens_seen



def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):

    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = metrics.calc_loss_batch_last_logits(input_batch, target_batch, model)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            examples_seen += input_batch.shape[0] # New: track examples instead of tokens 
            global_step += 1

            ## 130 batches: training, eval_Freq = 50 --> after 50 batches are processed in each epoch, we print train loss and val loss

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = metrics.evaluate_model(
                    model, train_loader, val_loader, loss_fn=metrics.calc_loss_batch_last_logits, num_batches=eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = metrics.calc_accuracy_loader(train_loader, model, num_batches=eval_iter)
        val_accuracy = metrics.calc_accuracy_loader(val_loader, model, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

def train(model, optimizer,  train_loader, val_loader, device, num_epochs, eval_freq, eval_iter):
    start_time = time.time()

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(model, train_loader, val_loader, optimizer, device,num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    
    return train_losses, val_losses, train_accs, val_accs, examples_seen


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,eval_freq, eval_iter, start_context, tokenizer):
    
    """
    Train a language model using the provided data loaders and optimizer.
    This function handles the complete training process including gradient updates,
    periodic evaluation, loss tracking, and sample text generation at each epoch.
    Steps: 
    - Set the model to training mode
    - Iterate over the training data loader
    - Compute the loss for each batch
    - Calculate gradients and update model weights
    - Track the total number of tokens processed
    - Perform evaluation at specified intervals
    - Generate sample text at the end

    Args:
        model: The neural language model to train
        train_loader: DataLoader containing batches of training data
        val_loader: DataLoader containing batches of validation data
        optimizer: Optimizer for updating model parameters
        device: Device to run computations on (CPU/GPU)
        num_epochs: Number of complete passes through the training dataset
        eval_freq: Frequency of evaluation steps (in global steps)
        eval_iter: Number of iterations to use during evaluation
        start_context: String used as a seed for text generation samples
        tokenizer: Tokenizer used for converting between text and token IDs
    Returns:
        tuple: (train_losses, val_losses, track_tokens_seen)
            - train_losses: List of training losses at evaluation points
            - val_losses: List of validation losses at evaluation points
            - track_tokens_seen: List of cumulative tokens processed at each evaluation point
    """

    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = metrics.calc_loss_batch(input_batch, target_batch, model)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel() # Returns the total number of elements (or tokens) in the input_batch.
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0: 
                train_loss, val_loss = metrics.evaluate_model(
                    model, train_loader, val_loader, loss_fn=metrics.calc_loss_batch, num_batches=eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f} "
                      f"Perplexity Train loss {torch.exp(torch.tensor(train_loss)):.3f}, Val loss {torch.exp(torch.tensor(val_loss)):.3f}")

        # Print a sample text after each epoch
        encoded = text_generator.text_to_token_ids(start_context, tokenizer).to(device)
        out_ids = text_generator._generate(
            idx=encoded,
            model=model,
            max_length=50

        )
        out_text = text_generator.token_ids_to_text(out_ids, tokenizer)
        print(out_text.replace("\n", " ")) # Print the generated text compact format 

    return train_losses, val_losses, track_tokens_seen

def train(model, train_loader, val_loader, optimizer, device, tokenizer,  num_epochs=10, eval_freq=5, eval_iter=0, start_context="Every effort moves you" ):
    
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

    model.to(device)
    
    train_losses, val_losses, tokens_seen = train_model( model, train_loader, val_loader, optimizer, device,num_epochs=num_epochs,eval_freq=eval_freq, eval_iter=eval_iter,start_context=start_context, tokenizer=tokenizer)


    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    return train_losses, val_losses, tokens_seen
