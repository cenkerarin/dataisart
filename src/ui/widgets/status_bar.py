"""
Status Bar Widget
================

Bottom status bar widget for displaying application status and information.
"""

from PyQt5.QtWidgets import (QStatusBar, QLabel, QProgressBar, QFrame, 
                             QHBoxLayout, QWidget)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import time


class StatusBar(QStatusBar):
    """Custom status bar for the application."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QStatusBar {
                background-color: #404040;
                color: #ffffff;
                border-top: 1px solid #555555;
                padding: 5px;
            }
            QLabel {
                color: #ffffff;
                padding: 2px 10px;
                border-right: 1px solid #666666;
            }
        """)
        
        # Initialize status elements
        self.setup_status_elements()
        
        # Timer for updating time and performance
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_time)
        self.update_timer.start(1000)  # Update every second
    
    def setup_status_elements(self):
        """Setup status bar elements."""
        
        # Main status message
        self.status_label = QLabel("Ready")
        self.status_label.setMinimumWidth(200)
        self.addWidget(self.status_label)
        
        # Separator
        self.addPermanentWidget(self.create_separator())
        
        # Camera status
        self.camera_status = QLabel("📷 Camera inactive")
        self.camera_status.setMinimumWidth(150)
        self.addPermanentWidget(self.camera_status)
        
        # Separator
        self.addPermanentWidget(self.create_separator())
        
        # Dataset status
        self.dataset_status = QLabel("📊 No dataset")
        self.dataset_status.setMinimumWidth(120)
        self.addPermanentWidget(self.dataset_status)
        
        # Separator
        self.addPermanentWidget(self.create_separator())
        
        # Performance indicator
        self.performance_label = QLabel("⚡ Performance: Good")
        self.performance_label.setMinimumWidth(150)
        self.addPermanentWidget(self.performance_label)
        
        # Separator
        self.addPermanentWidget(self.create_separator())
        
        # Time display
        self.time_label = QLabel()
        self.time_label.setMinimumWidth(80)
        self.addPermanentWidget(self.time_label)
        
        # Update time initially
        self.update_time()
    
    def create_separator(self):
        """Create a visual separator."""
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #666666;")
        return separator
    
    def update_status(self, message):
        """Update the main status message."""
        self.status_label.setText(message)
        
        # Auto-clear status after 5 seconds if it's not an error
        if not message.startswith("❌"):
            QTimer.singleShot(5000, lambda: self.status_label.setText("Ready"))
    
    def update_camera_status(self, status):
        """Update camera status."""
        self.camera_status.setText(status)
    
    def update_dataset_status(self, status):
        """Update dataset status."""
        self.dataset_status.setText(status)
    
    def update_performance_metrics(self):
        """Update performance metrics."""
        # Simple performance indicator based on time
        # In a real application, you might check CPU usage, memory, etc.
        performance_text = "⚡ Performance: Good"
        self.performance_label.setText(performance_text)
    
    def update_time(self):
        """Update the time display."""
        current_time = time.strftime("%H:%M:%S")
        self.time_label.setText(current_time) 