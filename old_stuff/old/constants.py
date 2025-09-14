"""
Global constants and device configuration for the election prediction project.

This module defines:
- Directory paths for data, models, results, predictions, and logs.
- The computing device (CPU, CUDA, MPS) to be used by PyTorch.
"""
import os
import torch

# --- File paths ---
DATA_DIR = "./data"
MODELS_DIR = "./models"
STUDY_DIR = "./results"
PREDS_DIR = "./preds"

