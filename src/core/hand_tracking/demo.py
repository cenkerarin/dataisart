#!/usr/bin/env python3
"""
Gesture Recognition Demo Runner
==============================

Simple script to run the gesture recognition demo without dealing with imports.
Run this from the hand_tracking directory: python demo.py
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the demo
from demos.gesture_classification_demo import main

if __name__ == "__main__":
    main() 