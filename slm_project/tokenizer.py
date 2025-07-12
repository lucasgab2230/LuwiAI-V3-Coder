import json

class TechnicalTextTokenizer:
    """
    A basic character-level tokenizer for technical texts.

    This tokenizer builds a vocabulary of all unique characters encountered
    during training and maps each character to an integer ID. It provides
    methods to encode text into a sequence of integer IDs and decode
    integer IDs back into text.

    Attributes:
    A basic character-level tokenizer for technical texts.
    """

    def __init__(self):
        self.char_to_int = {}
        self.int_to_char = {}
        self.vocab_size = 0

    def train(self, text):
        """
        Builds or updates the vocabulary based on the characters in the given text.

        The vocabulary includes all unique characters from the input text.
        If the tokenizer has been trained before, new characters will be added
        to the existing vocabulary.

        Args:
            text (str): The text to train the tokenizer on.
        """
        # Get unique characters from the input text and sort them
        new_unique_chars = sorted(list(set(text)))

        # Add new characters to the existing vocabulary
        current_vocab_size = len(self.char_to_int)
        for char in new_unique_chars:
            if char not in self.char_to_int:
                self.char_to_int[char] = current_vocab_size
                current_vocab_size += 1

        # Rebuild the integer-to-character mapping and update vocabulary size
        self.int_to_char = {i: ch for i, ch in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)
        print(f"Tokenizer vocabulary size: {self.vocab_size}")

    def encode(self, text):
        """
        Encodes a string into a list of integers.

        Characters not present in the vocabulary will be encoded as 0 (unknown character).

        Args:
            text (str): The string to encode.

        Returns:
            list: A list of integer IDs representing the encoded text.
        """
        # Map each character to its integer ID, using 0 for unknown characters
        return [self.char_to_int.get(c, 0) for c in text]

    def decode(self, tokens):
        """
        Decodes a list of integers back into a string.

        Args:
            tokens (list): A list of integer IDs to decode.
        """
        return "".join([self.int_to_char.get(i, '') for i in tokens])

    def save_vocabulary(self, filepath):
        """
        Saves the tokenizer's vocabulary to a JSON file.
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.char_to_int, f)
            print(f"Tokenizer vocabulary saved to {filepath}")
        except IOError as e:
            print(f"Error saving tokenizer vocabulary to {filepath}: {e}")

    def load_vocabulary(self, filepath):
        """
        Loads the tokenizer's vocabulary from a JSON file.

        Args:
            filepath (str): The path to the JSON file containing the vocabulary.

        Returns:
            bool: True if the vocabulary was loaded successfully, False otherwise.
        """
        try:
            with open(filepath, 'r') as f:
                self.char_to_int = json.load(f)

            # Rebuild the integer-to-character mapping
            self.int_to_char = {i: ch for ch, i in self.char_to_int.items()}
            self.vocab_size = len(self.char_to_int)
            print(f"Tokenizer vocabulary loaded from {filepath}. Vocabulary size: {self.vocab_size}")
            return True
        except FileNotFoundError:
            print(f"Error loading tokenizer vocabulary: File not found at {filepath}")
            return False
        except json.JSONDecodeError:
            print(f"Error loading tokenizer vocabulary: Invalid JSON format in {filepath}")
            return False
        except IOError as e:
            print(f"Error loading tokenizer vocabulary from {filepath}: {e}")
            return False
