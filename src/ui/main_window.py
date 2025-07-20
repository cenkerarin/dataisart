"""
Main Window - PyQt Desktop Application
=====================================

Main window for the hands-free data science application with PyQt interface.
Left panel: Data visualization and management
Right panel: Camera feed with hand gesture detection
"""

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QSplitter, QLabel, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

from .widgets.data_panel import DataPanel
from .widgets.camera_panel import CameraPanel
from .widgets.status_bar import StatusBar


class MainWindow(QMainWindow):
    """Main application window with data and camera panels."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠🖐️ Hands-Free Data Science")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set application theme
        self.setup_theme()
        
        # Initialize core components
        self.data_panel = None
        self.camera_panel = None
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
        """Setup the main user interface."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Create and add data panel (left side)
        self.data_panel = DataPanel()
        splitter.addWidget(self.data_panel)
        
        # Create and add camera panel (right side)
        self.camera_panel = CameraPanel()
        splitter.addWidget(self.camera_panel)
        
        # Set initial splitter sizes (60% data, 40% camera)
        splitter.setSizes([840, 560])
        
        # Setup status bar
        self.status_bar_widget = StatusBar()
        self.setStatusBar(self.status_bar_widget)
        
        # Connect signals
        self.setup_connections()
    
    def setup_connections(self):
        """Setup signal connections between components."""
        # Connect camera panel gesture signals to data panel
        if hasattr(self.camera_panel, 'gesture_detected'):
            self.camera_panel.gesture_detected.connect(self.data_panel.handle_gesture)
        
        # Connect data panel status updates to status bar
        if hasattr(self.data_panel, 'status_updated'):
            self.data_panel.status_updated.connect(self.status_bar_widget.update_status)
        
        # Connect camera panel status updates to status bar
        if hasattr(self.camera_panel, 'camera_status_changed'):
            self.camera_panel.camera_status_changed.connect(self.status_bar_widget.update_camera_status)
    
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