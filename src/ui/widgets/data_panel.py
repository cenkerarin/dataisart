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
                             QGroupBox, QProgressBar, QFileDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QIcon

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
        
        # Setup UI
        self.setup_ui()
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_data_display)
        self.update_timer.start(1000)  # Update every second
    
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
        
        # Gesture interaction area
        self.setup_gesture_area(layout)
        
        # Statistics
        self.setup_statistics(layout)
    
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
    
    def setup_gesture_area(self, layout):
        """Setup gesture interaction area."""
        gesture_group = QGroupBox("🖐️ Gesture Interactions")
        gesture_layout = QVBoxLayout(gesture_group)
        
        # Last gesture display
        self.last_gesture = QLabel("No gestures detected")
        self.last_gesture.setStyleSheet("""
            QLabel {
                background-color: #404040;
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
            }
        """)
        gesture_layout.addWidget(self.last_gesture)
        
        # Gesture log
        self.gesture_log = QTextEdit()
        self.gesture_log.setMaximumHeight(100)
        self.gesture_log.setReadOnly(True)
        self.gesture_log.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 5px;
            }
        """)
        gesture_layout.addWidget(self.gesture_log)
        
        layout.addWidget(gesture_group)
    
    def setup_statistics(self, layout):
        """Setup dataset statistics display."""
        stats_group = QGroupBox("📈 Dataset Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        # Statistics labels
        self.stats_rows = QLabel("Rows: 0")
        self.stats_columns = QLabel("Columns: 0")
        self.stats_memory = QLabel("Memory: 0 KB")
        
        for stat_label in [self.stats_rows, self.stats_columns, self.stats_memory]:
            stat_label.setStyleSheet("color: #cccccc; padding: 2px;")
            stats_layout.addWidget(stat_label)
        
        layout.addWidget(stats_group)
    
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
            
            # Update statistics
            self.stats_rows.setText(f"Rows: {shape[0]:,}")
            self.stats_columns.setText(f"Columns: {shape[1]:,}")
            
            memory_mb = self.current_dataset.memory_usage(deep=True).sum() / 1024 / 1024
            self.stats_memory.setText(f"Memory: {memory_mb:.1f} MB")
            
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
        """Handle gesture detection from camera panel."""
        try:
            gesture_name = gesture_data.get('gesture', 'unknown')
            confidence = gesture_data.get('confidence', 0.0)
            handedness = gesture_data.get('handedness', 'unknown')
            
            # Update last gesture display
            self.last_gesture.setText(
                f"🖐️ {handedness} hand: {gesture_name} ({confidence:.2f})"
            )
            
            # Add to gesture log
            log_entry = f"[{handedness}] {gesture_name} (conf: {confidence:.2f})"
            self.gesture_log.append(log_entry)
            
            # Emit signal for further processing
            self.gesture_received.emit(gesture_name, gesture_data)
            
            # Future: Add gesture-to-data interactions here
            # self.process_data_gesture(gesture_name, gesture_data)
            
        except Exception as e:
            self.status_updated.emit(f"❌ Gesture handling error: {str(e)}")
    
    def process_data_gesture(self, gesture_name, gesture_data):
        """Process gesture for data manipulation (placeholder for future implementation)."""
        # This is where you'll add gesture-to-data interaction logic
        # Examples:
        # - "point" gesture: select row/column
        # - "swipe_left/right": navigate through data
        # - "grab" gesture: select data range
        # - "pinch" gesture: zoom/filter data
        pass
    
    def update_data_display(self):
        """Periodic update of data display."""
        # Future: Add any real-time data updates here
        pass 