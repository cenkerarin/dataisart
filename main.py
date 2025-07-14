"""
Hands-Free Data Science - Main Application
==========================================

Entry point for the Streamlit application that enables hands-free data science
through gesture recognition and voice commands.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import core modules
from ui.pages.main_dashboard import MainDashboard
from core.data_processing.data_manager import DataManager
from core.hand_tracking.core.gesture_detector import GestureDetector
from core.voice_recognition.voice_handler import VoiceHandler
from core.llm_integration.ai_assistant import AIAssistant
from utils.helpers.config_loader import load_config

def main():
    """Main application entry point."""
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="Hands-Free Data Science",
        page_icon="🧠🖐️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load configuration
    config = load_config()
    
    # Initialize core components
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DataManager()
    
    if 'gesture_detector' not in st.session_state:
        st.session_state.gesture_detector = GestureDetector()
    
    if 'voice_handler' not in st.session_state:
        st.session_state.voice_handler = VoiceHandler()
    
    if 'ai_assistant' not in st.session_state:
        st.session_state.ai_assistant = AIAssistant(config)
    
    # Initialize main dashboard
    dashboard = MainDashboard(
        data_manager=st.session_state.data_manager,
        gesture_detector=st.session_state.gesture_detector,
        voice_handler=st.session_state.voice_handler,
        ai_assistant=st.session_state.ai_assistant
    )
    
    # Render the dashboard
    dashboard.render()

if __name__ == "__main__":
    main() 