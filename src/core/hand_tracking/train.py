#!/usr/bin/env python3
"""
Model Training Runner
====================

Simple script to run model training without dealing with imports.
Run this from the hand_tracking directory: python train.py
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the trainer
from tools.train_model import main

if __name__ == "__main__":
    main() 