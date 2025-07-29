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
        title = QLabel("📊 Data Panel")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Load data section (will be hidden when data is loaded)
        self.setup_load_section(layout)
        
        # Data display section
        self.setup_data_display(layout)
    
    def setup_load_section(self, layout):
        """Setup data loading section that disappears when data is loaded."""
        self.load_section = QGroupBox("Load Your Data")
        self.load_section.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: rgba(76, 175, 80, 0.1);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #4CAF50;
            }
        """)
        load_layout = QVBoxLayout(self.load_section)
        
        # Load dataset button - prominent and centered
        self.load_button = QPushButton("📁 Select Dataset File")
        self.load_button.clicked.connect(self.load_dataset)
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 15px 30px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
                transform: scale(1.02);
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        load_layout.addWidget(self.load_button)
        
        # Supported formats info
        formats_info = QLabel("Supports: CSV, Excel (.xlsx), JSON, Parquet files")
        formats_info.setStyleSheet("""
            color: #888888; 
            padding: 5px;
            font-style: italic;
            text-align: center;
        """)
        formats_info.setAlignment(Qt.AlignCenter)
        load_layout.addWidget(formats_info)
        
        layout.addWidget(self.load_section)
    
    def setup_data_display(self, layout):
        """Setup clean data display section."""
        # Data container (initially hidden)
        self.data_container = QFrame()
        self.data_container.setVisible(False)
        data_layout = QVBoxLayout(self.data_container)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.setSpacing(8)
        
        # Dataset info header (clean and minimal)
        self.dataset_header = QFrame()
        self.dataset_header.setStyleSheet("""
            QFrame {
                background-color: #3a3a3a;
                border-radius: 6px;
                padding: 5px;
            }
        """)
        header_layout = QHBoxLayout(self.dataset_header)
        header_layout.setContentsMargins(10, 8, 10, 8)
        
        self.dataset_info = QLabel("Dataset loaded")
        self.dataset_info.setStyleSheet("""
            color: #ffffff; 
            font-weight: bold;
            font-size: 14px;
        """)
        header_layout.addWidget(self.dataset_info)
        
        # Add new dataset button (small, right-aligned)
        self.new_dataset_btn = QPushButton("📁 Load Different Dataset")
        self.new_dataset_btn.clicked.connect(self.load_dataset)
        self.new_dataset_btn.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                border: none;
                color: white;
                padding: 6px 12px;
                font-size: 11px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        header_layout.addWidget(self.new_dataset_btn)
        
        data_layout.addWidget(self.dataset_header)
        
        # Data table (clean and focused)
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setStyleSheet("""
            QTableWidget {
                background-color: #2b2b2b;
                alternate-background-color: #323232;
                color: #ffffff;
                gridline-color: #444444;
                selection-background-color: #4CAF50;
                border: 1px solid #555555;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #404040;
                color: #ffffff;
                padding: 8px;
                border: 1px solid #555555;
                font-weight: bold;
            }
            QTableWidget::item {
                padding: 4px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: #4CAF50;
                color: white;
            }
        """)
        data_layout.addWidget(self.data_table)
        
        layout.addWidget(self.data_container)
    
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
                    self.load_section.setVisible(False) # Hide load section
                    self.data_container.setVisible(True) # Show data container
                else:
                    self.status_updated.emit("❌ Failed to load dataset")
                    
        except Exception as e:
            self.status_updated.emit(f"❌ Error loading dataset: {str(e)}")
    
    def update_dataset_display(self):
        """Update the dataset display with current data."""
        if self.current_dataset is None:
            return
        
        try:
            # Update dataset info with cleaner display
            shape = self.current_dataset.shape
            self.dataset_info.setText(f"📊 {shape[0]:,} rows × {shape[1]} columns")
            
            # Update data table (first 100 rows for performance)
            display_data = self.current_dataset.head(100)
            
            self.data_table.setRowCount(len(display_data))
            self.data_table.setColumnCount(len(display_data.columns))
            self.data_table.setHorizontalHeaderLabels([str(col) for col in display_data.columns])
            
            # Populate table with better formatting
            for i, row in enumerate(display_data.itertuples(index=False)):
                for j, value in enumerate(row):
                    # Format values nicely
                    if pd.isna(value):
                        display_value = "—"  # Better null display
                    elif isinstance(value, float):
                        display_value = f"{value:.3f}" if abs(value) < 1000 else f"{value:.2e}"
                    else:
                        display_value = str(value)
                    
                    item = QTableWidgetItem(display_value)
                    self.data_table.setItem(i, j, item)
            
            # Auto-resize columns but limit max width
            self.data_table.resizeColumnsToContents()
            for i in range(self.data_table.columnCount()):
                if self.data_table.columnWidth(i) > 200:
                    self.data_table.setColumnWidth(i, 200)
            
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
    
    def handle_voice_command(self, command_data):
        """Handle parsed voice commands from voice panel."""
        try:
            intent = command_data.get('intent', 'unknown')
            confidence = command_data.get('confidence', 0.0)
            
            self.status_updated.emit(f"🎤 Voice command: {intent} (confidence: {confidence:.2f})")
            
            # Handle different command intents
            if intent == 'show' or intent == 'describe':
                self.handle_show_command(command_data)
            elif intent == 'filter':
                self.handle_filter_command(command_data)
            elif intent == 'sort':
                self.handle_sort_command(command_data)
            elif intent == 'visualize':
                self.handle_visualize_command(command_data)
            elif intent == 'aggregate':
                self.handle_aggregate_command(command_data)
            elif intent == 'group':
                self.handle_group_command(command_data)
            else:
                self.status_updated.emit(f"⚠️ Voice command '{intent}' not yet implemented")
                
        except Exception as e:
            self.status_updated.emit(f"❌ Voice command error: {str(e)}")
    
    def handle_show_command(self, command_data):
        """Handle show/describe commands."""
        if self.current_dataset is None:
            self.status_updated.emit("❌ No dataset loaded")
            return
        
        # For now, just show basic dataset info
        rows, cols = self.current_dataset.shape
        self.status_updated.emit(f"📊 Dataset: {rows} rows, {cols} columns")
    
    def handle_filter_command(self, command_data):
        """Handle filter commands."""
        self.status_updated.emit("🔍 Filter command received - implementation coming soon")
        # TODO: Implement data filtering based on voice commands
    
    def handle_sort_command(self, command_data):
        """Handle sort commands."""
        self.status_updated.emit("📈 Sort command received - implementation coming soon")
        # TODO: Implement data sorting based on voice commands
    
    def handle_visualize_command(self, command_data):
        """Handle visualization commands."""
        self.status_updated.emit("📊 Visualization command received - implementation coming soon")
        # TODO: Implement data visualization based on voice commands
    
    def handle_aggregate_command(self, command_data):
        """Handle aggregation commands."""
        self.status_updated.emit("🔢 Aggregation command received - implementation coming soon")
        # TODO: Implement data aggregation based on voice commands
    
    def handle_group_command(self, command_data):
        """Handle group commands."""
        self.status_updated.emit("👥 Group command received - implementation coming soon")
        # TODO: Implement data grouping based on voice commands
    
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
                    # self.gesture_status.setText(f"⏳ {gesture_name.replace('_', ' ').title()} - starting...") # Removed gesture status
                    # self.gesture_status.setStyleSheet(""" # Removed gesture status
                    #     color: #FFC107; 
                    #     padding: 5px; 
                    #     background-color: #2d2d2d; 
                    #     border-radius: 3px; 
                    #     font-size: 12px;
                    #     font-weight: bold;
                    # """) # Removed gesture status
                    return
                
                # Check if we've held the gesture long enough to start continuous scrolling
                hold_duration = current_time - self.gesture_start_time
                if hold_duration >= self.gesture_hold_time:
                    # Start continuous scrolling
                    if not self.continuous_scrolling:
                        self.current_scroll_action = self.gesture_actions[gesture_name]
                        self.start_continuous_scrolling()
                        
                        self.status_updated.emit(f"▶️ Scrolling with {gesture_name} ({confidence:.2f})")
                        # confidence_bar = "█" * int(confidence * 10) # Removed confidence bar
                        # self.gesture_status.setText(f"▶️ {gesture_name.replace('_', ' ').title()} active {confidence_bar}") # Removed confidence bar
                        # self.gesture_status.setStyleSheet(""" # Removed confidence bar
                        #     color: #4CAF50; 
                        #     padding: 5px; 
                        #     background-color: #2d2d2d; 
                        #     border-radius: 3px; 
                        #     font-size: 12px;
                        #     font-weight: bold;
                        # """) # Removed confidence bar
                else:
                    # Show shorter countdown for faster response
                    remaining = self.gesture_hold_time - hold_duration
                    # self.gesture_status.setText(f"⏳ {gesture_name.replace('_', ' ').title()} - {remaining:.1f}s") # Removed gesture status
                    # self.gesture_status.setStyleSheet(""" # Removed gesture status
                    #     color: #FF9800; 
                    #     padding: 5px; 
                    #     background-color: #2d2d2d; 
                    #     border-radius: 3px; 
                    #     font-size: 12px;
                    #     font-weight: bold;
                    # """) # Removed gesture status
                    
            else:
                # Stop scrolling if confidence drops or gesture changes
                self.stop_continuous_scrolling()
                self.gesture_start_time = None
                self.last_executed_gesture = None
                
                # if confidence < self.gesture_confidence_threshold and hasattr(self, 'gesture_status'): # Removed gesture status
                #     if confidence > 0.5:  # Show low confidence warning # Removed gesture status
                #         self.gesture_status.setText(f"⚠️ Low confidence: {confidence:.2f}") # Removed gesture status
                #         self.gesture_status.setStyleSheet(""" # Removed gesture status
                #             color: #FF5722; 
                #             padding: 5px; 
                #             background-color: #2d2d2d; 
                #             border-radius: 3px; 
                #             font-size: 12px;
                #             font-weight: bold;
                #         """) # Removed gesture status
                #     else: # Removed gesture status
                #         self.gesture_status.setText("🤚 Ready for gestures") # Removed gesture status
                #         self.gesture_status.setStyleSheet(""" # Removed gesture status
                #             color: #4CAF50; 
                #             padding: 5px; 
                #             background-color: #2d2d2d; 
                #             border-radius: 3px; 
                #             font-size: 12px;
                #             font-weight: bold;
                #         """) # Removed gesture status
        
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
        """Reset gesture state (UI gestures removed but functionality preserved)."""
        # Stop any continuous scrolling
        self.stop_continuous_scrolling()
        
        # Reset gesture tracking
        self.gesture_start_time = None
        self.last_executed_gesture = None
    
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