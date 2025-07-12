import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import logging # Import the logging module
import os # Import os for checking checkpoint file existence

from .config import SLMConfig
from .model import SLMModel
from .dataset import TechnicalTextDataset
from .tokenizer import TechnicalTextTokenizer
from .utils.checkpoints import save_checkpoint, load_checkpoint
from .utils.logging_utils import setup_logging
from .cpu_optimizations import apply_cpu_optimizations

# Get the logger for this module
logger = logging.getLogger(__name__)

def train_model(config: SLMConfig):
    """
    Trains the Small Language Model.

    Args:
        config (SLMConfig): Configuration object for the training.
        This object contains hyperparameters, data paths, and optimization flags.
    """
    setup_logging() # Setup logging
    logger.info("Setting up training process...")

    # 1. Initialize Tokenizer
    tokenizer = TechnicalTextTokenizer()
    # In a real scenario, you would train the tokenizer on your dataset
    # For this example, we'll create a dummy vocabulary
    dummy_text = "This is a sample technical text for tokenizer training."
    tokenizer.train(dummy_text)
    vocab_size = len(tokenizer.vocab)
    logging.info(f"Tokenizer vocabulary size: {vocab_size}")
    logger.info(f"Tokenizer vocabulary size: {vocab_size}")

    # 2. Initialize Dataset and DataLoader
    # The TechnicalTextDataset handles loading and tokenizing data from specified file types.
    # The DataLoader provides an iterable over the dataset for batch processing.
    train_dataset = TechnicalTextDataset(config.train_data_path, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    logger.info(f"Training dataset size: {len(train_dataset)}")

    # 3. Initialize Model
    # The SLMModel is a simplified Transformer-based model.
    model = SLMModel(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads
    )

    # Apply CPU Optimizations (Placeholder)
    # apply_cpu_optimizations()

    # 4. Define Loss Function and Optimizer
    # CrossEntropyLoss is commonly used for language modeling tasks.
    # Adam is a popular optimization algorithm.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 5. Load Checkpoint to resume training if available
    start_epoch = 0
    logger.info(f"Checking for checkpoint at {config.checkpoint_path}")
    if os.path.exists(config.checkpoint_path):
        start_epoch = load_checkpoint(config.checkpoint_path, model, optimizer)
        logger.info(f"Resuming training from epoch {start_epoch + 1}")

    # Set the model to training mode
    model.train()
    logger.info("Starting training loop...")

    # 6. Training Loop
    for epoch in range(start_epoch, config.num_epochs):
        total_loss = 0
        logger.info(f"Epoch {epoch+1}/{config.num_epochs}")
        for batch_idx, data in enumerate(train_dataloader):
            # Move data to the appropriate device if using GPU (not in this CPU-only case, but good practice)
            # inputs, targets = data.to(device), targets.to(device)
            inputs, targets = data

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass: compute model output
            outputs = model(inputs)
            # Compute the loss (outputs are typically logits, targets are token IDs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # Optimizer step: update model parameters based on gradients
            optimizer.step()

            total_loss += loss.item()
            logging.debug(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch+1}/{config.num_epochs}, Average Loss: {avg_loss:.4f}")

        # Save Checkpoint (Placeholder)
        logging.info(f"Saving checkpoint for epoch {epoch+1} to {config.checkpoint_path}")
        logger.info(f"Saving checkpoint for epoch {epoch+1} to {config.checkpoint_path}")
        save_checkpoint(model, optimizer, epoch, config.checkpoint_path)

if __name__ == "__main__":
    # This block allows running the script directly for testing purposes.
    # Example usage: Create a dummy config and run training
    dummy_config = SLMConfig()
    dummy_config.train_data_path = "./data/sample_docs"  # Make sure this directory exists
    # Create a dummy data directory and file for testing
    import os
    os.makedirs(dummy_config.train_data_path, exist_ok=True)
    with open(os.path.join(dummy_config.train_data_path, "dummy_train.txt"), "w") as f:
        f.write("This is some sample training data.")

    train_model(dummy_config)
