#!/usr/bin/env python3
"""
Enhanced Data Collection Runner
===============================

Simple script to run enhanced data collection without import issues.
Run this from the hand_tracking directory: python run_enhanced_collection.py
"""

import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import and run the enhanced data collector
from tools.enhanced_data_collector import main

if __name__ == "__main__":
    print("🚀 Starting Enhanced Gesture Data Collection...")
    main() 