"""
Test Voice Integration
=====================

Test script to verify that voice recognition integration with gesture-based activation
is working properly in the main UI.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# Import main window
from ui.main_window import MainWindow

def main():
    """Test the voice integration."""
    app = QApplication(sys.argv)
    
    # Create main window
    window = MainWindow()
    window.show()
    
    print("🧪 Voice Integration Test")
    print("=" * 50)
    print("✅ Main window created with voice panel")
    print("📋 Test checklist:")
    print("   1. ✅ Voice panel should appear on the right side")
    print("   2. ✅ Camera panel should be in the center")
    print("   3. ✅ Data panel should be on the left")
    print("   4. 🎤 Test microphone button should work")
    print("   5. ✌️ Show peace sign gesture for 1 second to activate voice")
    print("   6. 🎯 Try voice commands like:")
    print("      - 'describe data'")
    print("      - 'filter age greater than 25'")
    print("      - 'sort by name ascending'")
    print("      - 'visualize column sales'")
    print()
    print("🚀 Starting application...")
    
    # Test timer to show status updates
    def show_test_status():
        if window.voice_panel:
            print("✅ Voice panel initialized")
            if hasattr(window.voice_panel, 'voice_handler') and window.voice_panel.voice_handler:
                print("✅ Voice handler ready")
            else:
                print("⚠️ Voice handler not ready - check dependencies")
        
        if window.camera_panel:
            print("✅ Camera panel initialized")
        
        print("🎯 Ready for testing!")
    
    QTimer.singleShot(2000, show_test_status)
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 