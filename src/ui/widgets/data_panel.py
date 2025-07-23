"""
Data Panel Widget
================

Left panel widget for data visualization and management.
Will display loaded datasets and allow gesture-based data manipulation.
"""

import pandas as pd
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QTableWidget, QTableWidgetItem, QPushButton,
                             QComboBox, QTextEdit, QSplitter, QFrame,
                             QGroupBox, QProgressBar, QFileDialog, QScrollBar)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QIcon
import time

import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

# Import core modules using correct path
from core.data_processing.data_manager import DataManager


class DataPanel(QFrame):
    """Data panel widget for dataset management and visualization."""
    
    # Signals
    status_updated = pyqtSignal(str)  # Status message signal
    gesture_received = pyqtSignal(str, dict)  # Gesture handling signal
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setMinimumWidth(400)
        
        # Initialize data manager
        self.data_manager = DataManager()
        self.current_dataset = None
        
        # Gesture-based scrolling system
        self.gesture_confidence_threshold = 0.75  # Lowered from 0.95 for better usability
        self.gesture_hold_time = 0.4  # Reduced from 1.0 seconds for faster response
        self.current_gesture_data = {}
        self.gesture_start_time = None
        self.last_executed_gesture = None
        self.scroll_smoothness = 8  # Increased scroll smoothness
        
        # Enhanced smoothing system
        self.continuous_scrolling = False
        self.scroll_timer = QTimer()
        self.scroll_timer.timeout.connect(self.execute_continuous_scroll)
        self.current_scroll_action = None
        self.scroll_speed_multiplier = 1.0
        
        # Gesture action mappings
        self.gesture_actions = {
            "pointing": self.scroll_down,
            "open_hand": self.scroll_right,
            "pinch": self.scroll_up,
            "fist": self.scroll_left
        }
        
        # Setup UI
        self.setup_ui()
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_data_display)
        self.update_timer.start(1000)  # Update every second
        
        # Setup gesture processing timer (faster updates for responsiveness)
        self.gesture_timer = QTimer()
        self.gesture_timer.timeout.connect(self.process_gesture_queue)
        self.gesture_timer.start(100)  # Check gestures every 100ms
    
    def setup_ui(self):
        """Setup the data panel user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("📊 Data Management")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Dataset controls
        self.setup_dataset_controls(layout)
        
        # Data preview
        self.setup_data_preview(layout)
    
    def setup_dataset_controls(self, layout):
        """Setup dataset loading and control widgets."""
        controls_group = QGroupBox("Dataset Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Load dataset button
        load_button = QPushButton("📁 Load Dataset")
        load_button.clicked.connect(self.load_dataset)
        load_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        controls_layout.addWidget(load_button)
        
        # Dataset info
        self.dataset_info = QLabel("No dataset loaded")
        self.dataset_info.setStyleSheet("color: #cccccc; padding: 5px;")
        controls_layout.addWidget(self.dataset_info)
        
        # Gesture control instructions
        gesture_instructions = QLabel(
            "🖐️ Gesture Controls:\n"
            "👆 Point → Scroll Down\n"
            "🤏 Pinch → Scroll Up\n"
            "✋ Open Hand → Scroll Right\n"
            "✊ Fist → Scroll Left\n"
            "(Need 75%+ confidence for 0.4s)"
        )
        gesture_instructions.setStyleSheet("""
            color: #888888; 
            padding: 8px; 
            background-color: #333333; 
            border-radius: 5px; 
            font-size: 11px;
            line-height: 1.3;
        """)
        controls_layout.addWidget(gesture_instructions)
        
        # Gesture status indicator
        self.gesture_status = QLabel("🤚 Ready for gestures")
        self.gesture_status.setStyleSheet("""
            color: #4CAF50; 
            padding: 5px; 
            background-color: #2d2d2d; 
            border-radius: 3px; 
            font-size: 12px;
            font-weight: bold;
        """)
        controls_layout.addWidget(self.gesture_status)
        
        layout.addWidget(controls_group)
    
    def setup_data_preview(self, layout):
        """Setup data preview table."""
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Data table
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setStyleSheet("""
            QTableWidget {
                background-color: #2b2b2b;
                alternate-background-color: #353535;
                color: #ffffff;
                gridline-color: #555555;
                selection-background-color: #4CAF50;
            }
            QHeaderView::section {
                background-color: #404040;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #555555;
            }
        """)
        preview_layout.addWidget(self.data_table)
        
        layout.addWidget(preview_group)
    

    
    def load_dataset(self):
        """Load a dataset file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open Dataset",
                "",
                "Data Files (*.csv *.xlsx *.json *.parquet);;All Files (*)"
            )
            
            if file_path:
                # Use data manager to load dataset
                success = self.data_manager.load_dataset(file_path)
                
                if success:
                    self.current_dataset = self.data_manager.current_dataset
                    self.update_dataset_display()
                    self.status_updated.emit(f"✅ Dataset loaded: {Path(file_path).name}")
                else:
                    self.status_updated.emit("❌ Failed to load dataset")
                    
        except Exception as e:
            self.status_updated.emit(f"❌ Error loading dataset: {str(e)}")
    
    def update_dataset_display(self):
        """Update the dataset display with current data."""
        if self.current_dataset is None:
            return
        
        try:
            # Update dataset info
            shape = self.current_dataset.shape
            self.dataset_info.setText(f"Dataset: {shape[0]} rows × {shape[1]} columns")
            
            # Update data table (first 100 rows)
            display_data = self.current_dataset.head(100)
            
            self.data_table.setRowCount(len(display_data))
            self.data_table.setColumnCount(len(display_data.columns))
            self.data_table.setHorizontalHeaderLabels([str(col) for col in display_data.columns])
            
            # Populate table
            for i, row in enumerate(display_data.itertuples(index=False)):
                for j, value in enumerate(row):
                    item = QTableWidgetItem(str(value))
                    self.data_table.setItem(i, j, item)
            
            # Auto-resize columns
            self.data_table.resizeColumnsToContents()
            
        except Exception as e:
            self.status_updated.emit(f"❌ Error updating display: {str(e)}")
    
    def handle_gesture(self, gesture_data):
        """Handle gesture detection from camera panel with confidence tracking."""
        try:
            gesture_name = gesture_data.get('gesture', 'unknown')
            confidence = gesture_data.get('confidence', 0.0)
            handedness = gesture_data.get('handedness', 'unknown')
            
            # Update current gesture data for processing
            self.current_gesture_data = {
                'gesture': gesture_name,
                'confidence': confidence,
                'handedness': handedness,
                'timestamp': time.time()
            }
            
            # Emit signal for further processing
            self.gesture_received.emit(gesture_name, gesture_data)
            
        except Exception as e:
            self.status_updated.emit(f"❌ Gesture handling error: {str(e)}")
    
    def process_gesture_queue(self):
        """Process gesture data with enhanced smoothness and continuous scrolling."""
        try:
            if not self.current_gesture_data or self.current_dataset is None:
                self.stop_continuous_scrolling()
                return
                
            gesture_name = self.current_gesture_data.get('gesture', 'unknown')
            confidence = self.current_gesture_data.get('confidence', 0.0)
            current_time = time.time()
            
            # Check if gesture is in our action mappings and meets confidence threshold
            if gesture_name in self.gesture_actions and confidence >= self.gesture_confidence_threshold:
                
                # Calculate scroll speed based on confidence (higher confidence = faster scroll)
                self.scroll_speed_multiplier = min(2.0, confidence / 0.75)  # Max 2x speed
                
                # If this is a new high-confidence gesture, start timing
                if (self.gesture_start_time is None or 
                    self.last_executed_gesture != gesture_name):
                    self.gesture_start_time = current_time
                    self.last_executed_gesture = gesture_name
                    self.stop_continuous_scrolling()  # Stop any previous scrolling
                    
                    self.status_updated.emit(f"🖐️ Detected {gesture_name} ({confidence:.2f}) - Starting...")
                    self.gesture_status.setText(f"⏳ {gesture_name.replace('_', ' ').title()} - starting...")
                    self.gesture_status.setStyleSheet("""
                        color: #FFC107; 
                        padding: 5px; 
                        background-color: #2d2d2d; 
                        border-radius: 3px; 
                        font-size: 12px;
                        font-weight: bold;
                    """)
                    return
                
                # Check if we've held the gesture long enough to start continuous scrolling
                hold_duration = current_time - self.gesture_start_time
                if hold_duration >= self.gesture_hold_time:
                    # Start continuous scrolling
                    if not self.continuous_scrolling:
                        self.current_scroll_action = self.gesture_actions[gesture_name]
                        self.start_continuous_scrolling()
                        
                        self.status_updated.emit(f"▶️ Scrolling with {gesture_name} ({confidence:.2f})")
                        confidence_bar = "█" * int(confidence * 10)  # Visual confidence indicator
                        self.gesture_status.setText(f"▶️ {gesture_name.replace('_', ' ').title()} active {confidence_bar}")
                        self.gesture_status.setStyleSheet("""
                            color: #4CAF50; 
                            padding: 5px; 
                            background-color: #2d2d2d; 
                            border-radius: 3px; 
                            font-size: 12px;
                            font-weight: bold;
                        """)
                else:
                    # Show shorter countdown for faster response
                    remaining = self.gesture_hold_time - hold_duration
                    self.gesture_status.setText(f"⏳ {gesture_name.replace('_', ' ').title()} - {remaining:.1f}s")
                    self.gesture_status.setStyleSheet("""
                        color: #FF9800; 
                        padding: 5px; 
                        background-color: #2d2d2d; 
                        border-radius: 3px; 
                        font-size: 12px;
                        font-weight: bold;
                    """)
                    
            else:
                # Stop scrolling if confidence drops or gesture changes
                self.stop_continuous_scrolling()
                self.gesture_start_time = None
                self.last_executed_gesture = None
                
                if confidence < self.gesture_confidence_threshold and hasattr(self, 'gesture_status'):
                    if confidence > 0.5:  # Show low confidence warning
                        self.gesture_status.setText(f"⚠️ Low confidence: {confidence:.2f}")
                        self.gesture_status.setStyleSheet("""
                            color: #FF5722; 
                            padding: 5px; 
                            background-color: #2d2d2d; 
                            border-radius: 3px; 
                            font-size: 12px;
                            font-weight: bold;
                        """)
                    else:
                        self.gesture_status.setText("🤚 Ready for gestures")
                        self.gesture_status.setStyleSheet("""
                            color: #4CAF50; 
                            padding: 5px; 
                            background-color: #2d2d2d; 
                            border-radius: 3px; 
                            font-size: 12px;
                            font-weight: bold;
                        """)
        
        except Exception as e:
            self.stop_continuous_scrolling()
            self.status_updated.emit(f"❌ Gesture processing error: {str(e)}")
    
    def start_continuous_scrolling(self):
        """Start continuous scrolling with current action."""
        if not self.continuous_scrolling and self.current_scroll_action:
            self.continuous_scrolling = True
            self.scroll_timer.start(50)  # 20 FPS for smooth scrolling
    
    def stop_continuous_scrolling(self):
        """Stop continuous scrolling."""
        if self.continuous_scrolling:
            self.continuous_scrolling = False
            self.scroll_timer.stop()
            self.current_scroll_action = None
    
    def execute_continuous_scroll(self):
        """Execute the current scroll action continuously."""
        if self.current_scroll_action and self.continuous_scrolling:
            self.current_scroll_action()
    
    def scroll_down(self):
        """Scroll down in the data table (pointing gesture)."""
        if not self.data_table or self.current_dataset is None:
            return
            
        vertical_scrollbar = self.data_table.verticalScrollBar()
        current_value = vertical_scrollbar.value()
        max_value = vertical_scrollbar.maximum()
        
        # Dynamic scroll amount based on confidence and speed multiplier
        scroll_amount = int(self.scroll_smoothness * self.scroll_speed_multiplier)
        new_value = min(current_value + scroll_amount, max_value)
        vertical_scrollbar.setValue(new_value)
    
    def scroll_up(self):
        """Scroll up in the data table (pinch gesture)."""
        if not self.data_table or self.current_dataset is None:
            return
            
        vertical_scrollbar = self.data_table.verticalScrollBar()
        current_value = vertical_scrollbar.value()
        
        # Dynamic scroll amount based on confidence and speed multiplier
        scroll_amount = int(self.scroll_smoothness * self.scroll_speed_multiplier)
        new_value = max(current_value - scroll_amount, 0)
        vertical_scrollbar.setValue(new_value)
    
    def scroll_right(self):
        """Scroll right in the data table (open_hand gesture)."""
        if not self.data_table or self.current_dataset is None:
            return
            
        horizontal_scrollbar = self.data_table.horizontalScrollBar()
        current_value = horizontal_scrollbar.value()
        max_value = horizontal_scrollbar.maximum()
        
        # Dynamic scroll amount for horizontal (slightly faster)
        scroll_amount = int(self.scroll_smoothness * 1.5 * self.scroll_speed_multiplier)
        new_value = min(current_value + scroll_amount, max_value)
        horizontal_scrollbar.setValue(new_value)
    
    def scroll_left(self):
        """Scroll left in the data table (fist gesture)."""
        if not self.data_table or self.current_dataset is None:
            return
            
        horizontal_scrollbar = self.data_table.horizontalScrollBar()
        current_value = horizontal_scrollbar.value()
        
        # Dynamic scroll amount for horizontal (slightly faster)
        scroll_amount = int(self.scroll_smoothness * 1.5 * self.scroll_speed_multiplier)
        new_value = max(current_value - scroll_amount, 0)
        horizontal_scrollbar.setValue(new_value)
    
    def process_data_gesture(self, gesture_name, gesture_data):
        """Process gesture for data manipulation (legacy method - now handled by process_gesture_queue)."""
        # This method is kept for compatibility but the main logic is now in process_gesture_queue
        pass
    
    def update_data_display(self):
        """Periodic update of data display."""
        # Future: Add any real-time data updates here
        pass
    
    def reset_gesture_status(self):
        """Reset gesture status indicator to ready state."""
        # Stop any continuous scrolling
        self.stop_continuous_scrolling()
        
        # Reset gesture tracking
        self.gesture_start_time = None
        self.last_executed_gesture = None
        
        if hasattr(self, 'gesture_status'):
            self.gesture_status.setText("🤚 Ready for gestures")
            self.gesture_status.setStyleSheet("""
                color: #4CAF50; 
                padding: 5px; 
                background-color: #2d2d2d; 
                border-radius: 3px; 
                font-size: 12px;
                font-weight: bold;
            """)
    
    def cleanup(self):
        """Clean up resources when panel is destroyed."""
        # Stop continuous scrolling
        self.stop_continuous_scrolling()
        
        # Stop all timers
        if hasattr(self, 'update_timer') and self.update_timer:
            self.update_timer.stop()
        
        if hasattr(self, 'gesture_timer') and self.gesture_timer:
            self.gesture_timer.stop()
        
        if hasattr(self, 'scroll_timer') and self.scroll_timer:
            self.scroll_timer.stop() 