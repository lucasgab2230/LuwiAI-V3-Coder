import os
import glob
import torch
from torch.utils.data import Dataset
from slm_project.tokenizer import TechnicalTextTokenizer

class TechnicalTextDataset(Dataset):
    def __init__(self, data_dir, tokenizer: TechnicalTextTokenizer, max_length=1024):
        """Initializes the TechnicalTextDataset.

        Loads text data from various file types in a specified directory,
        tokenizes it using the provided tokenizer, and splits it into sequences
        of a fixed maximum length.

        Args:
            data_dir (str): Path to the directory containing data files.
            tokenizer (TechnicalTextTokenizer): The tokenizer instance.
            max_length (int): Maximum sequence length for each training sample.
                              Input sequences will have length `max_length - 1`
                              and target sequences will have length `max_length - 1`.

        Raises:
            FileNotFoundError: If the specified data directory does not exist.
            ValueError: If no supported data files are found in the directory
                        or if no text is loaded.
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenized_data = []

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        self._load_and_tokenize_data()

    def _load_and_tokenize_data(self):
        all_text = ""
        for ext in ['*.txt', '*.md', '*.json', '*.jsonl']:
            for file_path in glob.glob(os.path.join(self.data_dir, ext)):
                try:
                    # Load text from supported file types
                    with open(file_path, 'r', encoding='utf-8') as f:
                        all_text += f.read() + "\n"  # Add newline to separate documents
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")

        if not all_text:
            raise ValueError(f"No text data loaded from the directory: {self.data_dir}. "
                             "Ensure it contains .txt, .md, .json, or .jsonl files.")

        # Train the tokenizer on the loaded data if it hasn't been trained
        # This is a basic approach; in a real scenario, the tokenizer
        # would likely be trained on a separate, larger corpus and saved.
        if not self.tokenizer.vocab:
             self.tokenizer.train(all_text)

        # Tokenize the concatenated text
        tokenized_all_text = self.tokenizer.encode(all_text)

        # Split tokenized data into sequences of max_length
        # We create sequences of size max_length, where the first max_length-1
        # tokens are the input and the last max_length-1 tokens (shifted) are the target.
        for i in range(0, len(tokenized_all_text) - self.max_length, self.max_length):
            self.tokenized_data.append(tokenized_all_text[i : i + self.max_length])

    def __len__(self):
        """Returns the number of sequences in the dataset."""
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        """Retrieves a single tokenized sequence pair (input and target).

        The input sequence is tokens [0] to [max_length-2], and the target
        sequence is tokens [1] to [max_length-1].

        Args:
            idx (int): The index of the sequence to retrieve.

        Returns:
            tuple: A tuple containing two torch.LongTensor:
                   - input_sequence: The input token sequence.
                   - target_sequence: The target token sequence (shifted).
        """
        return torch.tensor(self.tokenized_data[idx][:-1], dtype=torch.long), \
               torch.tensor(self.tokenized_data[idx][1:], dtype=torch.long)
