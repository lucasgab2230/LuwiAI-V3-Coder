import pytest
import torch
from slm_project.model import SLMModel
from slm_project.config import SLMConfig

class TestSLMModel:
    def test_model_initialization(self):
        config = SLMConfig()
        config.vocab_size = 100 # Set a vocab size for initialization test
        model = SLMModel(config)
        assert model is not None
        assert model.embedding.embedding_dim == config.embedding_dim
        assert model.transformer_blocks is not None
        assert len(model.transformer_blocks) == config.num_layers
        assert model.fc_out.out_features == config.vocab_size

        config_custom = SLMConfig(embedding_dim=128, num_layers=2, num_heads=4, vocab_size=2000)
        model_custom = SLMModel(config_custom)
        assert model_custom is not None
        assert model_custom.embedding.embedding_dim == config_custom.embedding_dim
        assert model_custom.transformer_blocks is not None
        assert len(model_custom.transformer_blocks) == config_custom.num_layers
        assert model_custom.fc_out.out_features == config_custom.vocab_size

    def test_model_forward_pass(self):
        config = SLMConfig(vocab_size=500, embedding_dim=32, num_layers=1, num_heads=1)
        model = SLMModel(config)
        
        batch_size = 8
        sequence_length = 64
        dummy_input = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
        
        output = model(dummy_input)
        assert output.shape == (batch_size, sequence_length, config.vocab_size)

        # Test with different batch size and sequence length
        batch_size = 16
        sequence_length = 32
        dummy_input = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
        
        output = model(dummy_input)
        assert output.shape == (batch_size, sequence_length, config.vocab_size)

        # Test with batch size 1
        batch_size = 1
        sequence_length = 100
        dummy_input = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
        
        output = model(dummy_input)
        assert output.shape == (batch_size, sequence_length, config.vocab_size)
