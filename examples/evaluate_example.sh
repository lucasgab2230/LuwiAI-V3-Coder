#!/bin/bash

# This script demonstrates how to run the evaluation script for the SLM project.

# Set the path to the trained model checkpoint
# Replace 'path/to/your/model_checkpoint.pth' with the actual path
MODEL_CHECKPOINT="/path/to/your/model_checkpoint.pth"

# Set the path to the evaluation data directory
EVAL_DATA_DIR="/path/to/your/evaluation_data/"

# Run the evaluation script
python /slm_project/evaluate.py --model_checkpoint "$MODEL_CHECKPOINT" --eval_data_dir "$EVAL_DATA_DIR"
