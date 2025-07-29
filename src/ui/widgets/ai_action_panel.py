"""
AI Action Panel Widget
======================

Panel widget to display AI-powered actions and results.
Shows voice recognition transcriptions, LLM analysis results, 
data insights, visualizations, and other AI outcomes.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QTextEdit, QFrame, QGroupBox, QScrollArea,
                             QSplitter, QPushButton, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, pyqtSlot
from PyQt5.QtGui import QFont, QTextCursor, QColor, QPalette

# Add paths for core modules
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

try:
    from core.llm_integration.ai_assistant import AIAssistant, ActionResult
    LLM_AVAILABLE = True
except ImportError:
    print("⚠️  LLM integration not available")
    LLM_AVAILABLE = False


class AIActionPanel(QFrame):
    """AI action panel for displaying AI-powered results and interactions."""
    
    # Signals
    action_requested = pyqtSignal(dict)  # Request action from AI
    status_updated = pyqtSignal(str)  # Status updates
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setMinimumHeight(300)
        self.setMaximumHeight(600)
        
        # AI assistant integration
        self.ai_assistant = None
        if LLM_AVAILABLE:
            try:
                # Basic config for AI assistant (can be expanded later)
                ai_config = {
                    "llm": {
                        "api_key": None,  # Will use environment variable if available
                        "model": "gpt-3.5-turbo",
                        "temperature": 0.7
                    }
                }
                self.ai_assistant = AIAssistant(ai_config)
                self.ai_assistant.initialize()
            except Exception as e:
                print(f"⚠️  Could not initialize AI assistant: {e}")
        
        # Action history
        self.action_history = []
        self.max_history = 50
        
        # Setup UI
        self.setup_ui()
        
        # Setup timer for UI updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(1000)  # Update every second
        
    def setup_ui(self):
        """Setup the unified AI action panel UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # Simple header - no status text
        title_label = QLabel("🤖 AI Panel")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setStyleSheet("color: #ffffff; padding: 8px;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create one unified scrollable content area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #555555;
                border-radius: 5px;
                background-color: #2a2a2a;
            }
            QScrollBar:vertical {
                background-color: #3a3a3a;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: #555555;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666666;
            }
        """)
        
        # Content widget for the scroll area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(12)
        
        # Voice Recognition Section (integrated)
        self.setup_voice_section(content_layout)
        
        # AI Analysis Section (integrated)
        self.setup_analysis_section(content_layout)
        
        # Action History Section (integrated)
        self.setup_history_section(content_layout)
        
        # Add stretch to push content to top
        content_layout.addStretch()
        
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

    def setup_voice_section(self, layout):
        """Setup integrated voice recognition section."""
        # Voice status indicator (compact)
        voice_header = QHBoxLayout()
        voice_icon = QLabel("🎤")
        voice_icon.setStyleSheet("font-size: 16px;")
        voice_header.addWidget(voice_icon)
        
        self.voice_status_label = QLabel("Voice: Inactive")
        self.voice_status_label.setStyleSheet("color: #cccccc; font-weight: bold; font-size: 12px;")
        voice_header.addWidget(self.voice_status_label)
        voice_header.addStretch()
        
        layout.addLayout(voice_header)
        
        # Voice transcription display (compact)
        self.voice_display = QTextEdit()
        self.voice_display.setMaximumHeight(60)  # Smaller height
        self.voice_display.setReadOnly(True)
        self.voice_display.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                padding: 4px;
            }
        """)
        self.voice_display.setPlaceholderText("Voice transcriptions...")
        layout.addWidget(self.voice_display)
        
        # Separator line
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setStyleSheet("color: #444444;")
        layout.addWidget(separator1)

    def setup_analysis_section(self, layout):
        """Setup integrated AI analysis section."""
        # AI Analysis header (compact)
        analysis_header = QHBoxLayout()
        analysis_icon = QLabel("🧠")
        analysis_icon.setStyleSheet("font-size: 16px;")
        analysis_header.addWidget(analysis_icon)
        
        analysis_label = QLabel("AI Analysis")
        analysis_label.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 12px;")
        analysis_header.addWidget(analysis_label)
        analysis_header.addStretch()
        
        # Progress bar for ongoing operations (initially hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(4)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 2px;
                background-color: #333333;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 2px;
            }
        """)
        analysis_header.addWidget(self.progress_bar)
        
        layout.addLayout(analysis_header)
        
        # Analysis display (main content area)
        self.analysis_display = QTextEdit()
        self.analysis_display.setReadOnly(True)
        self.analysis_display.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 3px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                line-height: 1.4;
                padding: 8px;
            }
        """)
        self.analysis_display.setPlaceholderText("AI analysis results will appear here...")
        layout.addWidget(self.analysis_display)
        
        # Separator line
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setStyleSheet("color: #444444;")
        layout.addWidget(separator2)

    def setup_history_section(self, layout):
        """Setup integrated action history section."""
        # History header with clear button
        history_header = QHBoxLayout()
        history_icon = QLabel("📋")
        history_icon.setStyleSheet("font-size: 16px;")
        history_header.addWidget(history_icon)
        
        history_label = QLabel("Action History")
        history_label.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 12px;")
        history_header.addWidget(history_label)
        
        history_header.addStretch()
        
        # Clear history button (small and integrated)
        clear_btn = QPushButton("Clear")
        clear_btn.setMaximumWidth(50)
        clear_btn.setMaximumHeight(20)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #444444;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 2px 6px;
                font-size: 9px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
        """)
        clear_btn.clicked.connect(self.clear_history)
        history_header.addWidget(clear_btn)
        
        layout.addLayout(history_header)
        
        # History display (compact)
        self.history_display = QTextEdit()
        self.history_display.setMaximumHeight(80)  # Smaller height
        self.history_display.setReadOnly(True)
        self.history_display.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #cccccc;
                border: 1px solid #444444;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 9px;
                padding: 4px;
            }
        """)
        self.history_display.setPlaceholderText("Action history...")
        layout.addWidget(self.history_display)

    # Remove the old create_* methods as they're no longer needed
    def create_voice_section(self):
        """Legacy method - now integrated into main UI"""
        pass
        
    def create_analysis_section(self):
        """Legacy method - now integrated into main UI"""
        pass
        
    def create_history_section(self):
        """Legacy method - now integrated into main UI"""
        pass
    
    @pyqtSlot(str)
    def update_voice_status(self, status):
        """Update voice recognition status."""
        self.voice_status_label.setText(f"Voice: {status}")
        if "listening" in status.lower():
            self.voice_status_label.setStyleSheet("color: #00ff00;")
        elif "processing" in status.lower():
            self.voice_status_label.setStyleSheet("color: #ffaa00;")
        else:
            self.voice_status_label.setStyleSheet("color: #cccccc;")
    
    @pyqtSlot(str)
    def update_voice_transcription(self, transcription):
        """Update voice transcription display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.voice_display.append(f"[{timestamp}] {transcription}")
        
        # Auto-scroll to bottom
        cursor = self.voice_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.voice_display.setTextCursor(cursor)
    
    @pyqtSlot(dict)
    def handle_voice_command(self, command_data):
        """Handle parsed voice command."""
        self.add_to_history(f"Voice Command: {command_data.get('command', 'Unknown')}")
        
        # Show processing
        self.show_processing("Processing voice command...")
        
        # If AI assistant is available, process the command
        if self.ai_assistant and LLM_AVAILABLE:
            try:
                result = self.ai_assistant.process_command(command_data)
                self.display_ai_result(result)
            except Exception as e:
                self.display_error(f"Error processing command: {str(e)}")
        else:
            self.display_message("Voice command received (AI processing not available)")
        
        self.hide_processing()
    
    def display_ai_result(self, result):
        """Display AI analysis result."""
        if not result:
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Format result based on type
        if hasattr(result, 'action_type'):
            self.analysis_display.append(f"\n[{timestamp}] AI Analysis:")
            self.analysis_display.append(f"Action: {result.action_type}")
            
            if hasattr(result, 'result') and result.result:
                self.analysis_display.append(f"Result: {result.result}")
                
            if hasattr(result, 'explanation') and result.explanation:
                self.analysis_display.append(f"Explanation: {result.explanation}")
                
            if hasattr(result, 'confidence') and result.confidence:
                self.analysis_display.append(f"Confidence: {result.confidence:.2f}")
        else:
            self.analysis_display.append(f"\n[{timestamp}] AI Result: {str(result)}")
        
        # Auto-scroll to bottom
        cursor = self.analysis_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.analysis_display.setTextCursor(cursor)
        
        self.add_to_history(f"AI Analysis completed")
    
    def display_message(self, message):
        """Display a general message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.analysis_display.append(f"\n[{timestamp}] {message}")
        
        # Auto-scroll to bottom
        cursor = self.analysis_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.analysis_display.setTextCursor(cursor)
    
    def display_error(self, error_message):
        """Display error message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.analysis_display.append(f"\n[{timestamp}] ERROR: {error_message}")
        
        # Auto-scroll to bottom
        cursor = self.analysis_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.analysis_display.setTextCursor(cursor)
        
        self.add_to_history(f"Error: {error_message}")
    
    def show_processing(self, message="Processing..."):
        """Show processing indicator."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        # self.status_label.setText(message) # Removed status_label
        # self.status_label.setStyleSheet("color: #ffaa00;") # Removed status_label
    
    def hide_processing(self):
        """Hide processing indicator."""
        self.progress_bar.setVisible(False)
        # self.status_label.setText("Ready") # Removed status_label
        # self.status_label.setStyleSheet("color: #00ff00;") # Removed status_label
    
    def add_to_history(self, action):
        """Add action to history."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.action_history.append(f"[{timestamp}] {action}")
        
        # Limit history size
        if len(self.action_history) > self.max_history:
            self.action_history = self.action_history[-self.max_history:]
        
        # Update display
        self.history_display.clear()
        self.history_display.setText("\n".join(self.action_history[-10:]))  # Show last 10
        
        # Auto-scroll to bottom
        cursor = self.history_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.history_display.setTextCursor(cursor)
    
    def clear_history(self):
        """Clear action history."""
        self.action_history.clear()
        self.history_display.clear()
        self.add_to_history("History cleared")
    
    def update_display(self):
        """Update display elements periodically."""
        # Update status if needed
        pass
    
    def cleanup(self):
        """Cleanup resources."""
        if self.update_timer:
            self.update_timer.stop()
        
        if self.ai_assistant:
            # Cleanup AI assistant if needed
            pass 