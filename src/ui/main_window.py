"""
Main Window - PyQt Desktop Application
=====================================

Main window for the hands-free data science application with PyQt interface.
Left panel: Data visualization and management
Upper right panel: Small camera feed with hand gesture detection
Lower right panel: AI action panel for voice recognition and LLM results
"""

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QSplitter, QLabel, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

from .widgets.data_panel import DataPanel
from .widgets.camera_panel import CameraPanel
from .widgets.voice_panel import VoicePanel
from .widgets.ai_action_panel import AIActionPanel
from .widgets.status_bar import StatusBar


class MainWindow(QMainWindow):
    """Main application window with data panel on left, webcam and AI panel on right."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠🖐️ Hands-Free Data Science")
        self.setGeometry(50, 50, 1800, 1200)  # Larger window for better panel coverage
        
        # Set application theme
        self.setup_theme()
        
        # Initialize core components
        self.data_panel = None
        self.camera_panel = None
        self.voice_panel = None  # Keep for compatibility but won't add to layout
        self.ai_action_panel = None
        self.status_bar_widget = None
        
        # Setup UI
        self.setup_ui()
        
        # Setup timer for updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_interface)
        self.update_timer.start(100)  # Update every 100ms
        
    def setup_theme(self):
        """Setup application theme and styling."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            
            QFrame {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 5px;
            }
            
            QLabel {
                color: #ffffff;
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
            
            QSplitter::handle {
                background-color: #555555;
                width: 3px;
            }
            
            QSplitter::handle:hover {
                background-color: #777777;
            }
        """)
    
    def setup_ui(self):
        """Setup the main user interface with new layout."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout with minimal margins to use full space
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)  # Reduced margins
        main_layout.setSpacing(5)  # Reduced spacing
        
        # Create main horizontal splitter (AI panel left, right panel right)
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Create and add AI action panel (left side - now prominent)
        self.ai_action_panel = AIActionPanel()
        self.ai_action_panel.setMinimumWidth(400)  # Ensure minimum usable width
        main_splitter.addWidget(self.ai_action_panel)
        
        # Create right panel container with vertical splitter for webcam and data panel
        right_splitter = QSplitter(Qt.Vertical)
        
        # Create and add camera panel (upper right - remove size constraints)
        self.camera_panel = CameraPanel()
        # Remove height constraints to allow proper sizing
        self.camera_panel.setMinimumHeight(200)  # Just minimal constraint
        right_splitter.addWidget(self.camera_panel)
        
        # Create and add data panel (lower right - moved from left)
        self.data_panel = DataPanel()
        self.data_panel.setMinimumHeight(200)  # Ensure minimum usable height
        right_splitter.addWidget(self.data_panel)
        
        # Set initial sizes for right panel using proportional values
        # Get total height and distribute proportionally
        total_height = 1200  # Updated window height
        camera_height = int(total_height * 0.45)  # 45% for camera
        data_height = int(total_height * 0.55)    # 55% for data
        right_splitter.setSizes([camera_height, data_height])
        
        # Add right panel to main splitter
        main_splitter.addWidget(right_splitter)
        
        # Set initial sizes for main splitter using proportional values
        # Get total width and distribute proportionally  
        total_width = 1800  # Updated window width
        ai_width = int(total_width * 0.6)      # 60% for AI chat
        right_width = int(total_width * 0.4)   # 40% for right panel
        main_splitter.setSizes([ai_width, right_width])
        
        # Create voice panel for compatibility (but don't add to layout)
        # This ensures existing connections still work
        self.voice_panel = VoicePanel()
        
        # Setup status bar
        self.status_bar_widget = StatusBar()
        self.setStatusBar(self.status_bar_widget)
        
        # Connect signals
        self.setup_connections()
    
    def setup_connections(self):
        """Setup signal connections between components."""
        # Connect camera panel gesture signals to data panel for gesture-based data control
        if hasattr(self.camera_panel, 'gesture_detected'):
            self.camera_panel.gesture_detected.connect(self.data_panel.handle_gesture)
            print("✅ Connected gesture detection to data panel")
        
        # Connect gesture detection to voice panel for gesture-triggered voice recognition
        # (Voice panel still handles the logic but AI panel displays results)
        if hasattr(self.camera_panel, 'gesture_detected') and hasattr(self.voice_panel, 'update_gesture_status'):
            self.camera_panel.gesture_detected.connect(self.handle_gesture_for_voice)
            print("✅ Connected gesture detection to voice panel")
        
        # Connect voice commands from voice panel to data panel for voice-controlled data operations
        if hasattr(self.voice_panel, 'command_parsed') and hasattr(self.data_panel, 'handle_voice_command'):
            self.voice_panel.command_parsed.connect(self.data_panel.handle_voice_command)
            print("✅ Connected voice commands to data panel")
        
        # NEW: Connect voice commands to AI action panel for display
        if hasattr(self.voice_panel, 'command_parsed') and hasattr(self.ai_action_panel, 'handle_voice_command'):
            self.voice_panel.command_parsed.connect(self.ai_action_panel.handle_voice_command)
            print("✅ Connected voice commands to AI action panel")
        
        # NEW: Connect voice status to AI action panel
        if hasattr(self.voice_panel, 'voice_status_changed') and hasattr(self.ai_action_panel, 'update_voice_status'):
            self.voice_panel.voice_status_changed.connect(self.ai_action_panel.update_voice_status)
            print("✅ Connected voice status to AI action panel")
        
        # NEW: Connect voice transcriptions to AI action panel
        if hasattr(self.voice_panel, 'transcription_received') and hasattr(self.ai_action_panel, 'update_voice_transcription'):
            self.voice_panel.transcription_received.connect(self.ai_action_panel.update_voice_transcription)
            print("✅ Connected voice transcriptions to AI action panel")
        
        # Connect voice status to status bar
        if hasattr(self.voice_panel, 'voice_status_changed'):
            self.voice_panel.voice_status_changed.connect(self.status_bar_widget.update_voice_status)
        
        # Connect data panel status updates to status bar
        if hasattr(self.data_panel, 'status_updated'):
            self.data_panel.status_updated.connect(self.status_bar_widget.update_status)
        
        # Connect camera panel status updates to status bar
        if hasattr(self.camera_panel, 'camera_status_changed'):
            self.camera_panel.camera_status_changed.connect(self.status_bar_widget.update_camera_status)
        
        # NEW: Connect AI action panel status to status bar
        if hasattr(self.ai_action_panel, 'status_updated'):
            self.ai_action_panel.status_updated.connect(self.status_bar_widget.update_status)
    
    def handle_gesture_for_voice(self, gesture_data):
        """Handle gesture data for voice panel and AI action panel."""
        if self.voice_panel and gesture_data:
            gesture_name = gesture_data.get("gesture", "")
            confidence = gesture_data.get("confidence", 0.0)
            
            # Update voice panel (for processing logic)
            self.voice_panel.update_gesture_status(gesture_name, confidence)
            
            # Also notify AI action panel for display
            if hasattr(self.ai_action_panel, 'add_to_history'):
                self.ai_action_panel.add_to_history(f"Gesture detected: {gesture_name} ({confidence:.2f})")
    
    def update_interface(self):
        """Update interface elements periodically."""
        # Update status bar with current information
        if self.status_bar_widget:
            self.status_bar_widget.update_performance_metrics()
    
    def closeEvent(self, event):
        """Handle application close event."""
        # Cleanup camera resources
        if self.camera_panel:
            self.camera_panel.cleanup()
        
        # Cleanup data panel resources  
        if self.data_panel:
            self.data_panel.cleanup()
        
        # Cleanup voice panel resources
        if self.voice_panel:
            self.voice_panel.cleanup()
        
        # Cleanup AI action panel resources
        if self.ai_action_panel:
            self.ai_action_panel.cleanup()
        
        # Stop update timer
        if self.update_timer:
            self.update_timer.stop()
        
        event.accept()


def create_application():
    """Create and return the PyQt application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Hands-Free Data Science")
    app.setApplicationVersion("1.0.0")
    
    # Set application icon (you can add an icon file later)
    # app.setWindowIcon(QIcon('path/to/icon.png'))
    
    return app


def main():
    """Main entry point for the PyQt application."""
    app = create_application()
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main()) 