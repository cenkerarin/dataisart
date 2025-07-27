"""
UI Integration Helpers
======================

Helper functions and classes for integrating AI assistant results with the PyQt UI.
Handles result display, visualization rendering, and user feedback.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTextEdit, QScrollArea, QFrame,
                             QGroupBox, QProgressBar, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor
from PyQt5.QtWebEngineWidgets import QWebEngineView
import plotly
import plotly.graph_objects as go

from .ai_assistant import ActionResult, ActionType

logger = logging.getLogger(__name__)

class AIResultDisplayWidget(QWidget):
    """Widget for displaying AI assistant results in the UI."""
    
    # Signals
    action_requested = pyqtSignal(str)  # When user requests new action
    visualization_ready = pyqtSignal(object)  # When visualization is ready
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.current_result = None
        
    def setup_ui(self):
        """Setup the result display UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title section
        self.title_label = QLabel("🤖 AI Assistant Results")
        self.title_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                color: #2196F3;
                padding: 10px;
                border: 2px solid #2196F3;
                border-radius: 8px;
                background-color: rgba(33, 150, 243, 0.1);
            }
        """)
        layout.addWidget(self.title_label)
        
        # Scroll area for results
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #555;
                border-radius: 5px;
            }
        """)
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        scroll.setWidget(self.content_widget)
        layout.addWidget(scroll)
        
        # Initial empty state
        self.show_empty_state()
        
    def show_empty_state(self):
        """Show empty state when no results are available."""
        self.clear_content()
        
        empty_label = QLabel("💬 Ready for voice commands!\n\nSay things like:\n• 'Describe data'\n• 'Visualize sales'\n• 'Analyze customer age'")
        empty_label.setAlignment(Qt.AlignCenter)
        empty_label.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 14px;
                padding: 40px;
                border: 2px dashed #555;
                border-radius: 10px;
                background-color: rgba(136, 136, 136, 0.1);
            }
        """)
        self.content_layout.addWidget(empty_label)
        
    def clear_content(self):
        """Clear all content from the display."""
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def display_result(self, result: ActionResult):
        """Display an AI assistant result."""
        self.current_result = result
        self.clear_content()
        
        # Create result frame
        result_frame = QFrame()
        result_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #555;
                border-radius: 8px;
                background-color: #3c3c3c;
                margin: 5px;
            }
        """)
        result_layout = QVBoxLayout(result_frame)
        
        # Status header
        status_header = self._create_status_header(result)
        result_layout.addWidget(status_header)
        
        # Message section
        if result.message:
            message_section = self._create_message_section(result.message)
            result_layout.addWidget(message_section)
        
        # Data section
        if result.data:
            data_section = self._create_data_section(result.data)
            result_layout.addWidget(data_section)
        
        # Visualization section
        if result.visualization:
            viz_section = self._create_visualization_section(result.visualization)
            result_layout.addWidget(viz_section)
        
        # Error section
        if result.error:
            error_section = self._create_error_section(result.error)
            result_layout.addWidget(error_section)
        
        # Suggestions section
        if result.suggestions:
            suggestions_section = self._create_suggestions_section(result.suggestions)
            result_layout.addWidget(suggestions_section)
        
        self.content_layout.addWidget(result_frame)
        self.content_layout.addStretch()
        
    def _create_status_header(self, result: ActionResult) -> QWidget:
        """Create status header widget."""
        header = QFrame()
        layout = QHBoxLayout(header)
        
        # Status icon and text
        if result.success:
            status_text = f"✅ {result.action_type.value.replace('_', ' ').title()}"
            color = "#4CAF50"
        else:
            status_text = f"❌ {result.action_type.value.replace('_', ' ').title()}"
            color = "#F44336"
        
        status_label = QLabel(status_text)
        status_label.setFont(QFont("Arial", 12, QFont.Bold))
        status_label.setStyleSheet(f"color: {color};")
        
        layout.addWidget(status_label)
        layout.addStretch()
        
        return header
    
    def _create_message_section(self, message: str) -> QWidget:
        """Create message display section."""
        section = QGroupBox("Result")
        layout = QVBoxLayout(section)
        
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 5px;
                font-size: 13px;
            }
        """)
        
        layout.addWidget(message_label)
        return section
    
    def _create_data_section(self, data: Dict[str, Any]) -> QWidget:
        """Create data display section."""
        section = QGroupBox("Data Analysis")
        layout = QVBoxLayout(section)
        
        # Create expandable data view
        data_text = QTextEdit()
        data_text.setMaximumHeight(200)
        data_text.setReadOnly(True)
        
        # Format data nicely
        formatted_data = self._format_data_for_display(data)
        data_text.setPlainText(formatted_data)
        
        data_text.setStyleSheet("""
            QTextEdit {
                background-color: rgba(0, 0, 0, 0.3);
                border: 1px solid #555;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        
        layout.addWidget(data_text)
        return section
    
    def _create_visualization_section(self, visualization) -> QWidget:
        """Create visualization display section."""
        section = QGroupBox("Visualization")
        layout = QVBoxLayout(section)
        
        # Create web view for Plotly
        web_view = QWebEngineView()
        web_view.setMinimumHeight(400)
        
        # Convert Plotly figure to HTML
        try:
            if hasattr(visualization, 'to_html'):
                html_content = visualization.to_html(include_plotlyjs='cdn')
            else:
                # Assume it's already JSON
                fig = go.Figure(json.loads(visualization))
                html_content = fig.to_html(include_plotlyjs='cdn')
            
            web_view.setHtml(html_content)
            layout.addWidget(web_view)
            
            # Emit signal that visualization is ready
            self.visualization_ready.emit(visualization)
            
        except Exception as e:
            logger.error(f"Error displaying visualization: {e}")
            error_label = QLabel(f"Error displaying visualization: {str(e)}")
            error_label.setStyleSheet("color: #F44336; padding: 10px;")
            layout.addWidget(error_label)
        
        return section
    
    def _create_error_section(self, error: str) -> QWidget:
        """Create error display section."""
        section = QGroupBox("Error")
        layout = QVBoxLayout(section)
        
        error_label = QLabel(error)
        error_label.setWordWrap(True)
        error_label.setStyleSheet("""
            QLabel {
                color: #F44336;
                padding: 10px;
                background-color: rgba(244, 67, 54, 0.1);
                border: 1px solid #F44336;
                border-radius: 5px;
            }
        """)
        
        layout.addWidget(error_label)
        return section
    
    def _create_suggestions_section(self, suggestions: List[str]) -> QWidget:
        """Create suggestions display section."""
        section = QGroupBox("Suggestions")
        layout = QVBoxLayout(section)
        
        for suggestion in suggestions:
            suggestion_item = QFrame()
            suggestion_item.setStyleSheet("""
                QFrame {
                    border: 1px solid #555;
                    border-radius: 5px;
                    background-color: rgba(255, 255, 255, 0.05);
                    margin: 2px;
                }
                QFrame:hover {
                    background-color: rgba(255, 255, 255, 0.1);
                    border-color: #777;
                }
            """)
            
            item_layout = QHBoxLayout(suggestion_item)
            
            suggestion_label = QLabel(f"💡 {suggestion}")
            suggestion_label.setStyleSheet("padding: 8px; color: #FFC107;")
            
            try_button = QPushButton("Try")
            try_button.setMaximumWidth(50)
            try_button.clicked.connect(lambda checked, s=suggestion: self.action_requested.emit(s))
            try_button.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
            """)
            
            item_layout.addWidget(suggestion_label)
            item_layout.addWidget(try_button)
            
            layout.addWidget(suggestion_item)
        
        return section
    
    def _format_data_for_display(self, data: Dict[str, Any]) -> str:
        """Format data dictionary for readable display."""
        try:
            if isinstance(data, dict) and "basic_info" in data:
                # Special formatting for dataset description
                output = []
                
                if "basic_info" in data:
                    output.append("=== DATASET OVERVIEW ===")
                    info = data["basic_info"]
                    output.append(f"Name: {info.get('name', 'Unknown')}")
                    output.append(f"Shape: {info.get('shape', 'Unknown')}")
                    output.append(f"Memory: {info.get('memory_usage', 'Unknown')}")
                    output.append("")
                
                if "column_info" in data and data["column_info"]:
                    output.append("=== COLUMNS ===")
                    for col, info in list(data["column_info"].items())[:5]:  # Limit to first 5
                        output.append(f"{col}:")
                        output.append(f"  Type: {info.get('type', 'Unknown')}")
                        output.append(f"  Non-null: {info.get('non_null_count', 0)}")
                        output.append(f"  Unique: {info.get('unique_values', 0)}")
                        if "sample_values" in info:
                            output.append(f"  Samples: {info['sample_values'][:3]}")
                        output.append("")
                
                if "missing_values" in data:
                    missing = {k: v for k, v in data["missing_values"].items() if v > 0}
                    if missing:
                        output.append("=== MISSING VALUES ===")
                        for col, count in missing.items():
                            output.append(f"{col}: {count}")
                
                return "\n".join(output)
            
            else:
                # Generic JSON formatting
                return json.dumps(data, indent=2, default=str)
                
        except Exception as e:
            return f"Error formatting data: {str(e)}"

class AICommandProcessor:
    """Processes AI commands and integrates with UI components."""
    
    def __init__(self, ai_assistant, voice_panel=None, data_panel=None):
        """Initialize with AI assistant and UI panels."""
        self.ai_assistant = ai_assistant
        self.voice_panel = voice_panel
        self.data_panel = data_panel
        
        # Connect signals if panels are available
        if self.voice_panel:
            self.voice_panel.command_parsed.connect(self.handle_voice_command)
    
    def handle_voice_command(self, parsed_command: Dict[str, Any]):
        """Handle a parsed voice command."""
        try:
            # Extract raw text from parsed command
            raw_text = parsed_command.get("raw_text", "")
            
            # Process with AI assistant
            result = self.ai_assistant.process_voice_command(raw_text, parsed_command)
            
            # Display result in UI
            if hasattr(self.voice_panel, 'result_display'):
                self.voice_panel.result_display.display_result(result)
            
            # Update data panel if visualization was created
            if result.success and result.visualization and self.data_panel:
                self.data_panel.display_visualization(result.visualization)
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling voice command: {e}")
            return None
    
    def process_text_command(self, command: str):
        """Process a text command (for testing or manual input)."""
        try:
            result = self.ai_assistant.process_voice_command(command)
            return result
        except Exception as e:
            logger.error(f"Error processing text command: {e}")
            return None

def create_ai_integration_widget(ai_assistant, parent=None) -> AIResultDisplayWidget:
    """Factory function to create AI integration widget."""
    widget = AIResultDisplayWidget(parent)
    
    # Example data for testing when no real data is available
    def show_example():
        from .ai_assistant import ActionResult, ActionType
        example_result = ActionResult(
            success=True,
            action_type=ActionType.DESCRIBE_DATA,
            message="This is an example result. Load a dataset and try voice commands!",
            suggestions=[
                "Load a CSV file first",
                "Try saying 'describe data'",
                "Use 'visualize column_name' for charts"
            ]
        )
        widget.display_result(example_result)
    
    # Show example initially
    QTimer.singleShot(1000, show_example)
    
    return widget 