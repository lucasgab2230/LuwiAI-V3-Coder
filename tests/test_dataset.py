import pytest
import os
import tempfile
import torch
from slm_project.dataset import TechnicalTextDataset
from slm_project.tokenizer import TechnicalTextTokenizer # Assuming a basic tokenizer for testing

@pytest.fixture
def dummy_data_dir():
    """Creates a temporary directory with dummy data files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "file1.txt"), "w") as f:
            f.write("This is a text file.")
        with open(os.path.join(tmpdir, "file2.md"), "w") as f:
            f.write("# This is a markdown file.")
        with open(os.path.join(tmpdir, "file3.json"), "w") as f:
            f.write('{"key": "value"}')
        with open(os.path.join(tmpdir, "file4.jsonl"), "w") as f:
            f.write('{"line1": "data"}\n{"line2": "data"}')
        yield tmpdir

def test_dataset_initialization_and_length(dummy_data_dir):
    """Tests if the dataset can be initialized and if __len__ returns a reasonable value."""
    tokenizer = TechnicalTextTokenizer()
    # Train the tokenizer on the dummy data
    tokenizer.train(dummy_data_dir)
    dataset = TechnicalTextDataset(dummy_data_dir, tokenizer)
    assert len(dataset) > 0
