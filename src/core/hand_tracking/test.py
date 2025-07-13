#!/usr/bin/env python3
"""
Test Runner
===========

Simple script to run tests without dealing with imports.
Run this from the hand_tracking directory: python test.py
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the test
from tests.test_data_collection import main

if __name__ == "__main__":
    main() 