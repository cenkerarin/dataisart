"""
Hand Tracking Module
===================

A comprehensive gesture recognition and hand tracking system using MediaPipe and ML.

This module provides:
- Hand detection and landmark extraction
- Gesture classification using machine learning
- Real-time gesture recognition demos
- Data collection tools for training
- Model training utilities

Structure:
    core/       - Core detection and classification functionality
    data/       - Data collection and management tools
    training/   - Model training utilities
    demos/      - Example applications and demos
    tests/      - Test files
"""

# Main exports for easy access
from .core.gesture_detector import GestureDetector
from .core.gesture_classifier import GestureClassifier
from .data.data_collector import GestureDataCollector

__version__ = "1.0.0"
__author__ = "DataIsArt Team"

# Make core classes easily accessible
__all__ = [
    "GestureDetector",
    "GestureClassifier", 
    "GestureDataCollector"
] 