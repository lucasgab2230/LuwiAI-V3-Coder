import torch

import os

from safetensors.torch import save_file, load_file
from typing import Optional, Dict, Any
from torch.nn import Module
from torch.optim import Optimizer


def save_checkpoint(
    model: Module, optimizer: Optimizer, epoch: int, checkpoint_path: str
):
    """
    Saves the model and optimizer state to a checkpoint file.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        epoch (int): The current epoch number.
        checkpoint_path (str): The absolute path to the checkpoint file.
    """
    # Ensure the directory for the checkpoint exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Create a dictionary to save the state. Safetensors requires string keys and values,
    # so we convert non-string values like epoch to string.
    state_dict: Dict[str, Any] = {
        "epoch": str(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # Add other relevant information here, e.g., loss, global step
    }

    # Safetensors requires tensor values
    # We can't save the optimizer state directly as it contains non-tensor values
    # A common practice is to save the model state and epoch.
    # If you need to save optimizer state, you would need to handle serialization differently.
    save_file(model.state_dict(), checkpoint_path)

    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path: str, model: Module, optimizer: Optimizer) -> Optional[int]:
    """
    Loads the model and optimizer state from a checkpoint file.

    Args:
        checkpoint_path: The absolute path to the checkpoint file.
        model: The model to load the state into.
        optimizer: The optimizer to load the state into.

    Returns:
        int or None: The epoch number from the checkpoint, or None if no checkpoint
                     was found or an error occurred during loading.
    """
    if not os.path.exists(checkpoint_path):
        # Log that no checkpoint was found
        print(f"No checkpoint found at {checkpoint_path}")
        return None

    try:
        # Load the checkpoint dictionary using safetensors
        state_dict = load_file(checkpoint_path)

        # Load the model state
        model.load_state_dict(state_dict)

        # Log successful loading
        print(f"Checkpoint loaded from {checkpoint_path}")

        # Note: Safetensors currently primarily supports saving tensors.
        # Saving and loading the optimizer state and epoch directly in safetensors
        # might require custom serialization or saving them separately.
        # This implementation focuses on loading the model state.
        # You might need to adjust this based on how epoch was saved, if at all.
        return None # Or logic to load epoch if it was saved

    except Exception as e:
        # Log any errors that occur during loading
        print(f"Error loading checkpoint from {checkpoint_path}: {e}")
        return None
