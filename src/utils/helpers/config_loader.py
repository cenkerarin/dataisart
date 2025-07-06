"""
Configuration Loader
====================

Utility to load and combine application configuration from various sources.
"""

import sys
from pathlib import Path

# Add config directory to path
config_path = Path(__file__).parent.parent.parent.parent / "config"
sys.path.insert(0, str(config_path))

from app.settings import *
from app.environment import Environment

def load_config():
    """
    Load complete application configuration by combining settings and environment.
    
    Returns:
        dict: Complete configuration dictionary
    """
    env = Environment()
    
    # Validate required environment variables
    missing_vars = env.validate_required_env_vars()
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Combine all configuration
    config = {
        "app": APP_CONFIG,
        "camera": {**CAMERA_CONFIG, **env.get_camera_config()},
        "hand_tracking": HAND_TRACKING_CONFIG,
        "voice": {**VOICE_CONFIG, **env.get_voice_config()},
        "llm": {
            **LLM_CONFIG,
            "api_key": env.openai_api_key,
            "org_id": env.openai_org_id,
        },
        "data": DATA_CONFIG,
        "visualization": VIZ_CONFIG,
        "environment": {
            "debug": env.debug,
            "log_level": env.log_level,
            "name": env.environment_name,
        },
        "paths": {
            "base_dir": BASE_DIR,
            "data_dir": DATA_DIR,
            "assets_dir": ASSETS_DIR,
            "logs_dir": LOGS_DIR,
        }
    }
    
    return config 