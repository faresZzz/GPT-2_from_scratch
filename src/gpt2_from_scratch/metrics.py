import torch 
import torch.nn as nn

def calc_loss_batch(input_batch, target_batch, model):
    # Set the model to evaluation mode
    training_mode = model.training
    model.eval()

    # Move the input and target to the device
    device = next(model.parameters()).device
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    
    # Forward pass
    logits = model(input_batch)

    # Calculate the loss
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    
    # Set the model back to its original mode
    model.train(training_mode)

    return loss

def calc_loss_batch_last_logits(input_batch, target_batch, model):
    # Set the model to evaluation mode
    training_mode = model.training
    model.eval()

    # Move the input and target to the device
    device = next(model.parameters()).device
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    
    # Forward pass
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
   
    # Calculate the loss
    loss = torch.nn.functional.cross_entropy(logits, target_batch, label_smoothing=0.1)
    
    # Set the model back to its original mode
    model.train(training_mode)

    return loss

def calc_loss_loader(data_loader, model, loss_fn, num_batches=None):
    # Set the model to evaluation mode
    training_mode = model.training
    model.eval()

    # Initialize the total loss
    total_loss = 0.

    # Determine the number of batches to evaluate
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))

    # Iterate over the data loader
    for i, (input_batch, target_batch) in enumerate(data_loader):

        # Calculate loss onnly for the first num_batches
        if i < num_batches:
            # Calculate the loss for the batch using provided function
            loss = loss_fn(input_batch, target_batch, model)

            # Accumulate the loss
            total_loss += loss.item()
        else:
            # Break the loop if the number of batches is reached
            break
    
    # Set the model back to its original mode
    model.train(training_mode)
    
    return total_loss / num_batches

def perplexity(loss):
    # Calculate perplexity from loss
    return torch.exp(torch.tensor(loss)).item()

def calc_accuracy_loader(data_loader, model, num_batches=None):
    # Set the model to evaluation mode
    training_mode = model.training
    model.eval()

    # Get the device
    device = next(model.parameters()).device

    # Initialize the number of correct predictions and the number of examples
    correct_predictions, num_examples = 0, 0

    # Determine the number of batches to evaluate
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    # Iterate over the data loader
    for i, (input_batch, target_batch) in enumerate(data_loader):
        # Calculate accuracy only for the first num_batches
        if i < num_batches:
            # Move the input and target to the device
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                # Inference of the model on the input batch 
                logits = model(input_batch)[:, -1, :]  # Logits of last output token
            
            # Get the predicted labels
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            # Count the number of correct predictions
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break

    # Set the model back to its original mode
    model.train(training_mode)

    return correct_predictions / num_examples

def evaluate_model(model, train_loader, val_loader, loss_fn, num_batches=None):
    # Set the model to evaluation mode
    training_mode = model.training
    model.eval()


    # Calculate the loss on the training and validation sets
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, loss_fn=loss_fn, num_batches=num_batches)
        val_loss = calc_loss_loader(val_loader, model, loss_fn=loss_fn, num_batches=num_batches)
    
    # Set the model back to its original mode
    model.train(training_mode)

    return train_loss, val_loss

def MMLUEval(model, val_loader, device, loss):
    pass
