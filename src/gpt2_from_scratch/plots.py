import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os


def plot_train_val_losses(epochs_seen, tokens_seen, train_losses, val_losses, labels={"x": "Epochs", "y": "Loss", "train":"Training loss", "validation":"Validation loss", "xlabel":"Tokens seen"}, save_path=None):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label=labels["train"])
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label=labels["validation"])
    
    # Set labels and legend
    ax1.set_xlabel(labels["x"])
    ax1.set_ylabel(labels["y"])
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel(labels["xlabel"])
    
    # Adjust layout to make room for the second x-axis
    fig.tight_layout() 

    # Save and show the plot
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()
    plt.close()
