"""
LLM Integration Module
=====================

This module provides AI-powered assistant capabilities for data analysis and visualization.
It integrates with voice recognition and provides a complete solution for hands-free data science.

Key Components:
- AIAssistant: Core AI assistant with data analysis actions
- VoiceAIBridge: Bridge between voice recognition and AI processing
- UIIntegration: UI components for displaying results
- Example App: Complete demonstration application

Usage:
    from src.core.llm_integration import AIAssistant, VoiceAIIntegration
    
    # Basic setup
    config = {"llm": {"api_key": "your-key", "model": "gpt-3.5-turbo"}}
    assistant = AIAssistant(config)
    assistant.initialize()
    
    # Load dataset and process commands
    assistant.load_dataset("data.csv")
    result = assistant.process_voice_command("describe data")
"""

from .ai_assistant import (
    AIAssistant,
    ActionResult,
    ActionType,
    VisualizationType,
    DatasetContext,
    DataAnalysisActions
)

from .ui_integration import (
    AIResultDisplayWidget,
    AICommandProcessor,
    create_ai_integration_widget
)

from .voice_ai_bridge import (
    VoiceAIBridge,
    VoiceAIIntegration
)

__version__ = "1.0.0"
__author__ = "Data Science Team"

__all__ = [
    # Core AI Assistant
    "AIAssistant",
    "ActionResult", 
    "ActionType",
    "VisualizationType",
    "DatasetContext",
    "DataAnalysisActions",
    
    # UI Integration
    "AIResultDisplayWidget",
    "AICommandProcessor", 
    "create_ai_integration_widget",
    
    # Voice Integration
    "VoiceAIBridge",
    "VoiceAIIntegration",
] 