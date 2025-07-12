import torch
import torch.nn.functional as F
import logging

from slm_project.model import SLMModel
from slm_project.dataset import TechnicalTextDataset
from slm_project.config import SLMConfig
from slm_project.utils.logging_utils import setup_logging

def evaluate_model(model: SLMModel, eval_dataset: TechnicalTextDataset, config: SLMConfig):
    """
    Evaluates the trained SLM model on the evaluation dataset.

    Args:
        model: The trained SLM model.
        eval_dataset: The evaluation dataset.
        config: The SLMConfig object.
    """
    setup_logging()
    logging.info("Starting model evaluation...")

    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():  # Disable gradient calculation
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.batch_size)
        for batch in eval_dataloader:
            inputs, targets = batch # Assuming dataset returns inputs and targets
            outputs = model(inputs)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

    average_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(average_loss)).item()
    logging.info(f"Evaluation Loss: {average_loss:.4f}")
    logging.info(f"Evaluation Perplexity: {perplexity:.4f}")
    logging.info("Model evaluation finished.")
