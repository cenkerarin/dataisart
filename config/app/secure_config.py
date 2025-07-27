"""
Secure Configuration Loader
============================

Handles sensitive configuration data securely using environment variables.
Never stores secrets in code or config files.
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SecureConfig:
    """Secure configuration loader for sensitive data."""
    
    @staticmethod
    def load_llm_config() -> Dict[str, Any]:
        """Load LLM configuration from environment variables."""
        api_key = os.getenv("OPENAI_API_KEY", "")
        
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. AI features will use pattern matching fallback.")
        
        return {
            "provider": "openai",
            "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            "api_key": api_key,
            "org_id": os.getenv("OPENAI_ORG_ID", ""),
            "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
            "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
            "timeout": int(os.getenv("OPENAI_TIMEOUT", "30")),
            "fallback_to_patterns": True,
        }
    
    @staticmethod
    def load_voice_config() -> Dict[str, Any]:
        """Load voice recognition configuration."""
        return {
            "recognition_timeout": int(os.getenv("VOICE_RECOGNITION_TIMEOUT", "5")),
            "phrase_timeout": float(os.getenv("PHRASE_TIMEOUT", "1.0")),
            "energy_threshold": int(os.getenv("ENERGY_THRESHOLD", "4000")),
            "dynamic_energy_threshold": True,
            "pause_threshold": float(os.getenv("PAUSE_THRESHOLD", "0.8")),
        }
    
    @staticmethod
    def load_camera_config() -> Dict[str, Any]:
        """Load camera configuration."""
        return {
            "width": int(os.getenv("CAMERA_WIDTH", "640")),
            "height": int(os.getenv("CAMERA_HEIGHT", "480")),
            "fps": int(os.getenv("CAMERA_FPS", "30")),
            "device_id": int(os.getenv("CAMERA_DEVICE_ID", "0")),
            "flip_horizontal": os.getenv("CAMERA_FLIP", "True").lower() == "true",
        }
    
    @staticmethod
    def load_data_config() -> Dict[str, Any]:
        """Load data processing configuration."""
        return {
            "supported_formats": [".csv", ".xlsx", ".json", ".parquet"],
            "max_file_size_mb": int(os.getenv("MAX_FILE_SIZE_MB", "100")),
            "sample_size_for_preview": int(os.getenv("SAMPLE_SIZE_FOR_PREVIEW", "1000")),
            "auto_detect_types": True,
            "cache_results": True,
        }
    
    @staticmethod
    def load_visualization_config() -> Dict[str, Any]:
        """Load visualization configuration."""
        return {
            "default_theme": os.getenv("DEFAULT_THEME", "plotly_dark"),
            "figure_width": int(os.getenv("FIGURE_WIDTH", "800")),
            "figure_height": int(os.getenv("FIGURE_HEIGHT", "600")),
            "export_formats": ["png", "pdf", "svg", "html"],
            "interactive": True,
        }
    
    @staticmethod
    def check_required_env_vars() -> Dict[str, str]:
        """Check for required environment variables and return missing ones."""
        required_vars = []
        missing_vars = {}
        
        # Optional but recommended variables
        recommended_vars = {
            "OPENAI_API_KEY": "Required for full AI features (can work without for basic pattern matching)"
        }
        
        for var, description in recommended_vars.items():
            if not os.getenv(var):
                missing_vars[var] = description
        
        return missing_vars
    
    @staticmethod
    def get_full_config() -> Dict[str, Any]:
        """Get complete configuration from environment variables."""
        return {
            "llm": SecureConfig.load_llm_config(),
            "voice": SecureConfig.load_voice_config(),
            "camera": SecureConfig.load_camera_config(),
            "data": SecureConfig.load_data_config(),
            "visualization": SecureConfig.load_visualization_config(),
        } 