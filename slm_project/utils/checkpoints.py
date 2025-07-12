import torch

import os

def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    """
    Saves the model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        checkpoint_path (str): The absolute path to the checkpoint file.
    """
    # Ensure the directory for the checkpoint exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Create a dictionary to save the state
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # You could add other relevant information here, e.g., loss, global step
    }, checkpoint_path)

    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Loads the model and optimizer state from a checkpoint file.

    Args:
        checkpoint_path (str): The absolute path to the checkpoint file.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.

    Returns:
        int or None: The epoch number from the checkpoint, or None if no checkpoint
                     was found or an error occurred during loading.
    """
    if not os.path.exists(checkpoint_path):
        # Log that no checkpoint was found
        print(f"No checkpoint found at {checkpoint_path}")
        return None

    try:
        # Load the checkpoint dictionary
        checkpoint = torch.load(checkpoint_path)

        # Load the model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Log successful loading
        print(f"Checkpoint loaded from {checkpoint_path}")

        # Return the epoch to resume training from
        return checkpoint.get('epoch', 0) # Use .get with a default for safety

    except Exception as e:
        # Log any errors that occur during loading
        print(f"Error loading checkpoint from {checkpoint_path}: {e}")
        return None
    print(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint['epoch']
