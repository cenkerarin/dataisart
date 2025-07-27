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

# LLM and AI Assistant settings
# Import secure configuration
try:
    from .secure_config import SecureConfig
    LLM_CONFIG = SecureConfig.load_llm_config()
except ImportError:
    # Fallback to basic configuration
    LLM_CONFIG = {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "temperature": 0.1,
        "max_tokens": 1000,
        "fallback_to_patterns": True,
    }

# Data analysis settings
DATA_CONFIG = {
    "supported_formats": [".csv", ".xlsx", ".json", ".parquet"],
    "max_file_size_mb": 100,
    "sample_size_for_preview": 1000,
    "auto_detect_types": True,
    "cache_results": True,
}

# Visualization settings
VISUALIZATION_CONFIG = {
    "default_theme": "plotly_dark",
    "figure_width": 800,
    "figure_height": 600,
    "export_formats": ["png", "pdf", "svg", "html"],
    "interactive": True,
} 