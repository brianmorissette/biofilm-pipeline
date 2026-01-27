#!/bin/bash

# The partition we want (short==24 hours, long=7 days)
#SBATCH --partition=short
# One node
#SBATCH -N 1
# One job on that node
#SBATCH -n 1
# Request 4 CPU Cores
#SBATCH -c 4
# Please give me a GPU
#SBATCH --gres=gpu:2
# Ask for memory
#SBATCH --mem=64gb

# Get a node for more general use.

# Run a python program using our local virtual environment
cd /home/bfmorissette/biofilm-pipeline
/home/bfmorissette/.local/bin/uv run -- wandb agent brianmorissette-worcester-polytechnic-institute/biofilm-pipeline-sweep-v4-full-data/hm9kv42l --count 50