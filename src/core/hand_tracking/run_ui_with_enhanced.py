#!/usr/bin/env python3
"""
UI Test with Enhanced Classifier
================================

Test script to run the UI with the enhanced gesture classifier
without making any changes to the UI code itself.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import and run the main UI
from src.ui.main_window import MainWindow
from PyQt5.QtWidgets import QApplication

def main():
    """Run the UI with enhanced gesture recognition."""
    print("🚀 Starting UI with Enhanced Gesture Recognition...")
    print("✨ The enhanced classifier is now integrated seamlessly!")
    
    app = QApplication(sys.argv)
    
    # Create and show the main window
    main_window = MainWindow()
    main_window.show()
    
    print("📱 UI launched successfully!")
    print("💡 The gesture classifier will now use the enhanced model automatically")
    
    # Start the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 