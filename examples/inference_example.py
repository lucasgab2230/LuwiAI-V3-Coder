import torch
import sys

# Add the project directory to the Python path
sys.path.append('.')

from slm_project.model import SLMModel
from slm_project.tokenizer import TechnicalTextTokenizer
from slm_project.config import SLMConfig
from slm_project.utils.checkpoints import load_checkpoint

def load_trained_model(config, checkpoint_path, device):
    """Loads a trained model from a checkpoint."""
    model = SLMModel(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads
    ).to(device)

    # A dummy optimizer is needed for load_checkpoint, but its state won't be used for inference
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
    print(f"Loaded model from epoch {start_epoch}")
    model.eval() # Set model to evaluation mode
    return model

def generate_text(model, tokenizer, prompt, max_length=100, device='cpu'):
    """Generates text using the trained model."""
    model.eval()
    input_tokens = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)

    generated_tokens = input_tokens

    for _ in range(max_length - len(input_tokens)):
        with torch.no_grad():
            output = model(input_tensor)
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            generated_tokens.append(next_token)
            input_tensor = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0).to(device)
            if next_token == tokenizer.eos_token_id: # Assuming an EOS token is implemented in tokenizer
                break

    return tokenizer.decode(generated_tokens)

if __name__ == "__main__":
    # Example Usage:
    # This is a simplified example. In a real scenario, you'd load
    # the config and checkpoint path from command line arguments or a config file.
    config = SLMConfig() # Load your configuration
    checkpoint_path = "/checkpoints/latest_checkpoint.pt" # Specify the path to your trained model checkpoint

    # For inference on CPU
    device = torch.device("cpu")

    try:
        # Load the trained model
        model = load_trained_model(config, checkpoint_path, device)

        # Initialize the tokenizer (needs to be trained on the same data as the model)
        # For this example, we'll create a dummy tokenizer and train it on a simple string.
        # In a real application, you would load a trained tokenizer.
        tokenizer = TechnicalTextTokenizer()
        dummy_train_text = "This is some technical text for tokenizer training."
        tokenizer.train(dummy_train_text)

        # Generate text
        prompt = "The main function in Python is defined using"
        generated_text = generate_text(model, tokenizer, prompt, max_length=200, device=device)

        print("Generated Text:")
        print(generated_text)

    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}. Please train the model first.")
    except Exception as e:
        print(f"An error occurred: {e}")
