import os

class SLMConfig:
    """
    Configuration class for the Small Language Model project.
    """
    def __init__(self):
        self.embedding_dim = 256
        """Dimension of the token embeddings and the internal representation."""
        self.num_layers = 4
        """Number of transformer layers in the model."""
        self.num_heads = 4
        """Number of attention heads in each transformer layer."""

        self.learning_rate = 1e-4
        """The learning rate for the optimizer during training."""
        self.batch_size = 32
        """The number of samples per batch during training and evaluation."""
        self.num_epochs = 10
        """The total number of training epochs."""

        self.train_data_path = os.path.join("data", "sample_docs") # Default path
        """Path to the directory containing training data files (.txt, .md, .json, .jsonl)."""
        self.eval_data_path = os.path.join("data", "sample_docs") # Default path
        """Path to the directory containing evaluation data files (.txt, .md, .json, .jsonl)."""

        self.use_int8_quantization = False
        """Boolean flag to enable or disable INT8 quantization for model optimization."""

    def load_from_file(self, filepath: str):
        """
        Loads configuration from a JSON or YAML file.

        Args:
            filepath: The path to the configuration file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file format is not supported or invalid.
            ImportError: If required libraries (json or PyYAML) are not installed.
        """
        # This is a placeholder implementation.
        # In a real scenario, you would detect file type (e.g., based on extension)
        # and use the appropriate library (e.g., `json` or `PyYAML`) to load the data.
        # You would then iterate over the loaded dictionary and update the attributes
        # of this SLMConfig object, ensuring type compatibility and handling potential errors.
        print(f"Placeholder: Loading configuration from {filepath}")
        print("Requires installing 'PyYAML' for YAML support.")
        # Example outline:
        # import json # or import yaml
        # with open(filepath, 'r') as f:
        #     config_data = json.load(f) # or yaml.safe_load(f)
        # for key, value in config_data.items():
        #     if hasattr(self, key):
        #         setattr(self, key, value)
        #     else:
        #         print(f"Warning: Unknown configuration key in file: {key}")
