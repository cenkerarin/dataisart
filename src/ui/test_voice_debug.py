"""
Voice Recognition Debug Test
============================

Specialized test application for debugging voice recognition issues.
Shows detailed information about transcription and command parsing.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget, QSplitter
from PyQt5.QtCore import QTimer, Qt

# Import our debug panel
from ui.widgets.voice_debug_panel import VoiceDebugPanel
from ui.widgets.camera_panel import CameraPanel

class VoiceDebugWindow(QMainWindow):
    """Main window for voice recognition debugging."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔧 Voice Recognition Debug Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
        """)
        
        # Initialize components
        self.voice_debug_panel = None
        self.camera_panel = None
        
        # Setup UI
        self.setup_ui()
        
        # Setup connections
        self.setup_connections()
        
        print("🔧 Voice Recognition Debug Tool")
        print("=" * 50)
        print("This tool helps you debug voice recognition issues:")
        print("1. 📊 Real-time audio level monitoring")
        print("2. 🎤 Detailed transcription logging")
        print("3. 🤖 Command parsing analysis")
        print("4. 📈 Success rate statistics")
        print("5. ⚙️ Adjustable silence threshold")
        print()
        print("🚀 Debug window ready!")
    
    def setup_ui(self):
        """Setup the debug UI."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Create camera panel (left side)
        self.camera_panel = CameraPanel()
        self.camera_panel.setMaximumWidth(600)
        splitter.addWidget(self.camera_panel)
        
        # Create voice debug panel (right side)
        self.voice_debug_panel = VoiceDebugPanel()
        self.voice_debug_panel.setMinimumWidth(500)
        splitter.addWidget(self.voice_debug_panel)
        
        # Set initial splitter sizes (50% camera, 50% voice debug)
        splitter.setSizes([600, 600])
    
    def setup_connections(self):
        """Setup signal connections."""
        # Connect gesture detection to voice panel
        if hasattr(self.camera_panel, 'gesture_detected') and hasattr(self.voice_debug_panel, 'update_gesture_status'):
            self.camera_panel.gesture_detected.connect(self.handle_gesture_for_voice)
            print("✅ Connected gesture detection to voice debug panel")
        
        # Timer to show status updates
        QTimer.singleShot(2000, self.show_debug_status)
    
    def handle_gesture_for_voice(self, gesture_data):
        """Handle gesture data for voice debug panel."""
        if self.voice_debug_panel and gesture_data:
            gesture_name = gesture_data.get("gesture", "")
            confidence = gesture_data.get("confidence", 0.0)
            self.voice_debug_panel.update_gesture_status(gesture_name, confidence)
    
    def show_debug_status(self):
        """Show debug status information."""
        print("\n🔍 Debug Status Check:")
        
        if self.voice_debug_panel:
            if hasattr(self.voice_debug_panel, 'voice_handler') and self.voice_debug_panel.voice_handler:
                print("✅ Voice handler initialized")
            else:
                print("⚠️ Voice handler not ready")
            
            if hasattr(self.voice_debug_panel, 'command_parser') and self.voice_debug_panel.command_parser:
                print("✅ Command parser initialized")
            else:
                print("⚠️ Command parser not ready")
        
        if self.camera_panel:
            print("✅ Camera panel ready")
        
        print("\n📋 Debug Instructions:")
        print("1. Click 'Start Camera' to enable gesture detection")
        print("2. Use 'Test Mic' to verify microphone is working")
        print("3. Try 'Test Parse' to verify command parsing")
        print("4. Show ✌️ peace sign for 1 second to activate voice")
        print("5. Speak test commands:")
        print("   - 'describe data'")
        print("   - 'filter age greater than 25'")
        print("   - 'sort by name ascending'")
        print("6. Watch the debug logs for detailed information")
        print("\n🎯 Ready for voice debugging!")
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Cleanup components
        if self.camera_panel:
            self.camera_panel.cleanup()
        
        if self.voice_debug_panel:
            self.voice_debug_panel.cleanup()
        
        event.accept()

def main():
    """Run the voice debug application."""
    app = QApplication(sys.argv)
    
    # Create and show debug window
    debug_window = VoiceDebugWindow()
    debug_window.show()
    
    # Test command parsing immediately
    def test_parsing():
        if debug_window.voice_debug_panel:
            debug_window.voice_debug_panel.test_command_parsing()
    
    QTimer.singleShot(3000, test_parsing)
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 