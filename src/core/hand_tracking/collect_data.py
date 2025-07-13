#!/usr/bin/env python3
"""
Data Collection Runner
=====================

Simple script to run data collection without dealing with imports.
Run this from the hand_tracking directory: python collect_data.py
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the data collector
from tools.data_collector import main

if __name__ == "__main__":
    main() 