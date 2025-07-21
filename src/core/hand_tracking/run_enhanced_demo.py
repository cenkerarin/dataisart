#!/usr/bin/env python3
"""
Enhanced Gesture Demo Runner
=============================

Simple script to run enhanced gesture demo without import issues.
Run this from the hand_tracking directory: python run_enhanced_demo.py
"""

import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import and run the enhanced demo
from demos.enhanced_gesture_demo import main

if __name__ == "__main__":
    print("🎬 Starting Enhanced Gesture Recognition Demo...")
    main() 