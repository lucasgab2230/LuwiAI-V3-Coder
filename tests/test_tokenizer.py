import pytest
from slm_project.tokenizer import TechnicalTextTokenizer

def test_tokenizer_train():
    tokenizer = TechnicalTextTokenizer()
    text = "This is a test sentence. It includes some special characters: .,!?"
    tokenizer.train(text)
    # Check if all unique characters in the text are in the vocabulary
    unique_chars = set(text)
    assert len(tokenizer.char_to_id) == len(unique_chars)
    for char in unique_chars:
        assert char in tokenizer.char_to_id
    assert len(tokenizer.id_to_char) == len(tokenizer.char_to_id)

    # Test with another text to see if vocabulary updates
    text_2 = "Another test with different chars: <>[]{}"
    tokenizer.train(text_2)
    unique_chars_2 = set(text + text_2)
    assert len(tokenizer.char_to_id) == len(unique_chars_2)
    for char in unique_chars_2:
        assert char in tokenizer.char_to_id
    assert len(tokenizer.id_to_char) == len(tokenizer.char_to_id)

    # Test with empty string
    tokenizer = TechnicalTextTokenizer()
    tokenizer.train("")
    for char in unique_chars:
        assert char in tokenizer.char_to_id
    assert len(tokenizer.id_to_char) == len(tokenizer.char_to_id)

def test_tokenizer_encode_decode():
    tokenizer = TechnicalTextTokenizer()
    text = "Hello world! This is a test."
    tokenizer.train(text + " .,!?") # Train with a larger vocabulary
    tokenizer.train(text)
    encoded_tokens = tokenizer.encode(text) # type: ignore
    decoded_text = tokenizer.decode(encoded_tokens)
    assert decoded_text == text

    # Test with empty string
    encoded_empty = tokenizer.encode("")
    decoded_empty = tokenizer.decode(encoded_empty)
    assert decoded_empty == ""

    # Test with string of special characters
    special_chars_text = ".,!?"
    tokenizer.train(special_chars_text)
    encoded_special = tokenizer.encode(special_chars_text)
    decoded_special = tokenizer.decode(encoded_special)
    assert decoded_special == special_chars_text

    # Test with a long string
    long_text = "A" * 1000
    tokenizer.train(long_text)
    assert tokenizer.decode(tokenizer.encode(long_text)) == long_text

def test_tokenizer_technical_text():
    tokenizer = TechnicalTextTokenizer()