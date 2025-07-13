"""
Hand Tracking Module
===================

A comprehensive gesture recognition and hand tracking system using MediaPipe and machine learning.

This module provides:
- Real-time hand detection and tracking
- ML-based gesture classification
- Data collection tools
- Training utilities
- Demo applications

Quick Start:
-----------
from src.core.hand_tracking import GestureDetector, GestureClassifier

# Initialize detector
detector = GestureDetector()
detector.initialize({"max_num_hands": 1})

# Initialize classifier
classifier = GestureClassifier()
classifier.train()

# Use them together
results = detector.detect_hands(frame)
if results["hands_detected"]:
    gesture = classifier.predict_gesture(results["hands"][0]["landmarks"])
    print(f"Detected gesture: {gesture['gesture']}")
"""

# Import core classes
from .core.gesture_detector import GestureDetector
from .core.gesture_classifier import GestureClassifier

# Export main classes
__all__ = [
    "GestureDetector",
    "GestureClassifier"
]

# Version information
__version__ = "1.0.0"
__author__ = "Hand Tracking Team"
__description__ = "Real-time hand tracking and gesture recognition system"
