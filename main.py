"""
Hands-Free Data Science - Main Application
==========================================

Entry point for the hands-free data science PyQt desktop application with 
gesture recognition and voice commands.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import UI module
from ui import main as ui_main

def main():
    """Main application entry point."""
    
    print("🧠🖐️ Starting Hands-Free Data Science Application...")
    print("=" * 50)
    
    # Start PyQt desktop application
    return ui_main()

if __name__ == "__main__":
    sys.exit(main())