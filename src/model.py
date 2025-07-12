import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Simplified Multi-Head Self-Attention layer.

    This module implements a basic multi-head self-attention mechanism
    as used in Transformer models. It projects the input into queries,
    keys, and values, computes attention scores, and aggregates values
    based on these scores.

    Args:
        embed_dim (int): The dimension of the input and output embeddings.
        num_heads (int): The number of attention heads. embed_dim must be
                         divisible by num_heads.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.all_head_dim = self.head_dim * num_heads
        # Ensure embed_dim is divisible by num_heads
        if self.all_head_dim != embed_dim:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Linear layers for projecting input to queries, keys, and values
        self.qkv_proj = nn.Linear(embed_dim, 3 * self.all_head_dim)
        # Linear layer for projecting the concatenated attention outputs
        self.out_proj = nn.Linear(self.all_head_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass of the self-attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (batch_size, num_heads, seq_len, head_dim)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        # Multiply weights by values to get the attention output
        attention_output = torch.matmul(attention_weights, v) # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate the attention outputs from different heads
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.reshape(batch_size, seq_len, self.all_head_dim)
        # Project the concatenated output
        output = self.out_proj(attention_output)

        return output

class FeedForward(nn.Module):
    """A simple two-layer feed-forward network with GELU activation.

    This module is part of the Transformer block and processes the output
    of the self-attention layer.

    Args:
        embed_dim (int): The dimension of the input and output embeddings.
        inner_dim (int): The dimension of the hidden layer.
    """
    def __init__(self, embed_dim, inner_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, inner_dim)
        self.gelu = nn.GELU() # GELU is generally performant on CPU
        self.fc2 = nn.Linear(inner_dim, embed_dim)
        # Note: Using GELU is a common practice in modern transformers.
        # While less common on very old hardware, PyTorch's implementation
        # is generally efficient on CPU. If performance is critical on
        # Ivy Bridge, ReLU could be considered as an alternative, but GELU
        # often provides better model performance.

    def forward(self, x):
        """
        Forward pass of the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        return self.fc2(self.gelu(self.fc1(x)))

class TransformerBlock(nn.Module):
    """A single Transformer block consisting of self-attention and feed-forward layers.

    Each Transformer block applies a self-attention mechanism followed by a
    feed-forward network. Residual connections and layer normalization are
    applied around both sub-layers.

    Args:
        embed_dim (int): The dimension of the input and output embeddings.
        num_heads (int): The number of attention heads in the self-attention layer.
        ff_inner_dim (int): The dimension of the inner layer in the feed-forward network.
    """
    def __init__(self, embed_dim, num_heads, ff_inner_dim):
        super().__init__()
        self.attention = SelfAttention(embed_dim, num_heads)
        # Layer normalization before the attention layer
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, ff_inner_dim)
        # Layer normalization before the feed-forward layer
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass of the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        # Apply self-attention with residual connection and layer normalization
        x = x + self.attention(self.norm1(x))
        # Apply feed-forward network with residual connection and layer normalization
        x = x + self.feed_forward(self.norm2(x))
        # Note: The order of operations (norm -> attention/ff -> residual) is pre-norm,
        # which is common in many modern Transformer variants.
        return x

class SLMModel(nn.Module):
    """
    Simplified Small Language Model based on the Transformer architecture.
    """
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_inner_dim=None):
        """
        Args:
            vocab_size (int): The size of the vocabulary.
            embed_dim (int): The dimension of the token and position embeddings.
            num_layers (int): The number of Transformer blocks.
            num_heads (int): The number of attention heads in each Transformer block.
            ff_inner_dim (int, optional): The inner dimension of the feed-forward networks.
                                         Defaults to 4 * embed_dim.
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Assuming a fixed maximum sequence length for positional encoding.
        # A more advanced approach could use rotary embeddings or learnable positions.
        self.position_embedding = nn.Embedding(1024, embed_dim)
        # Create a list of Transformer blocks
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ff_inner_dim if ff_inner_dim is not None else 4 * embed_dim) for _ in range(num_layers)])
        # Output projection layer to map embeddings back to vocabulary size
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_ids):
        """
        Forward pass of the SLM model.

        Args:
            token_ids (torch.Tensor): Input tensor of token IDs, shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output logits for the next token prediction, shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = token_ids.size()
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=token_ids.device)
        embeddings = self.token_embedding(token_ids) + self.position_embedding(pos_ids)
        for layer in self.layers:
            embeddings = layer(embeddings)
        logits = self.output_proj(embeddings)
        return logits
