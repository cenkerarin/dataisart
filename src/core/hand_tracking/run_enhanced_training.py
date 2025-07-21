#!/usr/bin/env python3
"""
Enhanced Model Training Runner
==============================

Simple script to run enhanced model training without import issues.
Run this from the hand_tracking directory: python run_enhanced_training.py
"""

import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import and run the enhanced trainer
from tools.enhanced_trainer import main

if __name__ == "__main__":
    print("🤖 Starting Enhanced Gesture Model Training...")
    main() 