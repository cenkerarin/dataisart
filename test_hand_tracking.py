"""
Hand Tracking Test - Entry Point
================================

Simple entry point to run the hand tracking test dashboard.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Configure Streamlit page
st.set_page_config(
    page_title="Hand Tracking Test Dashboard",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import and run the dashboard
from ui.pages.hand_tracking_test import main

if __name__ == "__main__":
    main() 