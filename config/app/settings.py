"""
Application Settings Configuration
=================================

Central configuration file for the Hands-Free Data Science application.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
LOGS_DIR = BASE_DIR / "logs"

# Application settings
APP_CONFIG = {
    "name": "Hands-Free Data Science",
    "version": "1.0.0",
    "description": "Interact with datasets using hands and voice",
    "author": "Data Science Team",
    "debug": os.getenv("DEBUG", "False").lower() == "true",
}

# Camera and gesture settings
CAMERA_CONFIG = {
    "width": 640,
    "height": 480,
    "fps": 30,
    "device_id": 0,
    "flip_horizontal": True,
}

# Hand tracking settings
HAND_TRACKING_CONFIG = {
    "max_num_hands": 2,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5,
    "static_image_mode": False,
}

# Voice recognition settings
VOICE_CONFIG = {
    "recognition_timeout": 5,
    "phrase_timeout": 1,
    "energy_threshold": 4000,
    "dynamic_energy_threshold": True,
    "pause_threshold": 0.8,
}

# LLM settings
LLM_CONFIG = {
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 1000,
    "timeout": 30,
}

# Data processing settings
DATA_CONFIG = {
    "max_file_size_mb": 100,
    "supported_formats": [".csv", ".xlsx", ".json", ".parquet"],
    "max_display_rows": 1000,
    "max_display_columns": 50,
}

# Visualization settings
VIZ_CONFIG = {
    "default_theme": "plotly_white",
    "figure_width": 800,
    "figure_height": 600,
    "color_palette": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
} 