import re

def normalize_whitespace(text: str) -> str:
    """
    Normalizes whitespace in a string by replacing multiple spaces with a single space
    and standardizing newline characters.
    """
    text = re.sub(r'[ \\t]+', ' ', text)  # Replace multiple spaces/tabs with a single space
    text = re.sub(r'[\r\n]+', '\n', text) # Standardize newlines
    return text.strip() # Remove leading/trailing whitespace

# Add other text cleaning functions here as needed
# def remove_comments(text: str) -> str:
#    pass # Placeholder
import pytest
import torch
import os
import logging

from slm_project.utils.logging_utils import setup_logging
from slm_project.utils.checkpoints import save_checkpoint, load_checkpoint
# from slm_project.utils.text_cleaner import clean_technical_text # Uncomment when implemented

@pytest.fixture
def cleanup_logging_file():
    log_file = "test_training.log"
    yield
    if os.path.exists(log_file):
        os.remove(log_file)

@pytest.fixture
def cleanup_checkpoint_file():
    checkpoint_file = "test_checkpoint.pth"
    yield
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

def test_setup_logging(cleanup_logging_file):
    log_file = "test_training.log"
    logger = setup_logging(log_file)
    assert isinstance(logger, logging.Logger)

    # Test if handlers are set up (console and file)
    assert any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers)
    assert any(isinstance(handler, logging.FileHandler) and handler.baseFilename.endswith(log_file) for handler in logger.handlers)

    # Test logging to file
    log_message = "Test log message."
    logger.info(log_message)
    with open(log_file, "r") as f:
        log_content = f.read()
    assert log_message in log_content

def test_save_and_load_checkpoint(cleanup_checkpoint_file):
    checkpoint_file = "test_checkpoint.pth"
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch = 5

    # Save checkpoint
    save_checkpoint(model, optimizer, epoch, checkpoint_file)
    assert os.path.exists(checkpoint_file)

    # Create new model and optimizer for loading
    model_loaded = torch.nn.Linear(10, 2)
    optimizer_loaded = torch.optim.Adam(model_loaded.parameters(), lr=0.001)

    # Load checkpoint
    loaded_epoch = load_checkpoint(checkpoint_file, model_loaded, optimizer_loaded)
    assert loaded_epoch == epoch

    # Check if model and optimizer states are loaded
    assert all(torch.equal(p1, p2) for p1, p2 in zip(model.state_dict().values(), model_loaded.state_dict().values()))
    assert str(optimizer.state_dict()) == str(optimizer_loaded.state_dict()) # Comparing string representation for optimizer state

