"""
Environment Configuration
========================

Handles environment variables and runtime configuration.
"""

import os
from typing import Optional

class Environment:
    """Environment configuration manager."""
    
    def __init__(self):
        """Initialize environment configuration."""
        self.load_environment()
    
    def load_environment(self):
        """Load environment variables."""
        # Try to load from .env file if it exists
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment."""
        return os.getenv("OPENAI_API_KEY")
    
    @property
    def openai_org_id(self) -> Optional[str]:
        """Get OpenAI organization ID from environment."""
        return os.getenv("OPENAI_ORG_ID")
    
    @property
    def debug(self) -> bool:
        """Get debug mode from environment."""
        return os.getenv("DEBUG", "False").lower() == "true"
    
    @property
    def log_level(self) -> str:
        """Get log level from environment."""
        return os.getenv("LOG_LEVEL", "INFO").upper()
    
    @property
    def environment_name(self) -> str:
        """Get environment name."""
        return os.getenv("ENV", "development")
    
    def get_camera_config(self) -> dict:
        """Get camera configuration from environment."""
        return {
            "device_id": int(os.getenv("CAMERA_DEVICE_ID", "0")),
            "width": int(os.getenv("CAMERA_WIDTH", "640")),
            "height": int(os.getenv("CAMERA_HEIGHT", "480")),
        }
    
    def get_voice_config(self) -> dict:
        """Get voice recognition configuration from environment."""
        return {
            "recognition_timeout": int(os.getenv("VOICE_RECOGNITION_TIMEOUT", "5")),
            "energy_threshold": int(os.getenv("VOICE_ENERGY_THRESHOLD", "4000")),
        }
    
    def validate_required_env_vars(self) -> list:
        """Validate that required environment variables are set."""
        missing_vars = []
        
        if not self.openai_api_key:
            missing_vars.append("OPENAI_API_KEY")
        
        return missing_vars 