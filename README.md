## Small Language Model (SLM) for Technical Text on Intel Ivy Bridge CPUs

This project implements a Small Language Model (SLM) specifically designed and optimized for running on older Intel Ivy Bridge CPUs (x86-64 architecture) with support for SSE 4.2 and AVX 1.0, but *without* relying on AVX2 instructions. The model is tailored for processing technical texts, such as code, documentation (Markdown), and structured data (JSON, JSONL).

### Goals

The primary goals of this project are:

- To create a functional Small Language Model capable of understanding and generating technical text.
- To achieve reasonable inference and training performance on legacy Intel CPU hardware (Ivy Bridge and similar architectures).
- To provide an open-source, modular, and well-documented codebase for others to learn from and contribute to.
- To explore and implement CPU-specific optimizations, particularly INT8 quantization, without requiring advanced instruction sets like AVX2.

### Key Features

- **Simplified Transformer Architecture:** A lightweight implementation of the Transformer model.
- **CPU Optimizations:** Includes `torch.set_num_threads` and a framework for INT8 quantization tailored for older CPUs.
- **Technical Text Compatibility:** Designed to handle various technical document formats (.txt, .md, .json, .jsonl).
- **Custom Tokenizer:** A tokenizer adapted for the nuances of technical language, including code and markup.
- **Trainable:** Supports training on custom technical datasets.
- **Checkpointing:** Allows saving and resuming training progress.
- **Evaluation Metrics:** Tracks training progress using loss and perplexity.
- **Modular Pipeline:** Separate components for data loading, tokenization, model definition, training, and evaluation.

### Installation

1. Clone the repository:
