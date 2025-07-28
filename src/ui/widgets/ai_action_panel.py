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
        """Setup the AI action panel UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Header
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #4a4a4a;
                border: 1px solid #666666;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        header_layout = QHBoxLayout(header_frame)
        
        title_label = QLabel("🤖 AI Action Panel")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setStyleSheet("color: #ffffff; border: none; padding: 0px;")
        header_layout.addWidget(title_label)
        
        # Status indicator
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #00ff00; border: none; padding: 0px;")
        header_layout.addWidget(self.status_label)
        
        main_layout.addWidget(header_frame)
        
        # Create splitter for different sections
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # Voice Recognition Section
        self.voice_section = self.create_voice_section()
        splitter.addWidget(self.voice_section)
        
        # AI Analysis Section
        self.analysis_section = self.create_analysis_section()
        splitter.addWidget(self.analysis_section)
        
        # Action History Section
        self.history_section = self.create_history_section()
        splitter.addWidget(self.history_section)
        
        # Set initial sizes (30% voice, 40% analysis, 30% history)
        splitter.setSizes([120, 160, 120])
        
    def create_voice_section(self):
        """Create voice recognition display section."""
        group_box = QGroupBox("🎤 Voice Recognition")
        group_box.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        layout = QVBoxLayout(group_box)
        
        # Voice status
        self.voice_status_label = QLabel("Status: Inactive")
        self.voice_status_label.setStyleSheet("color: #cccccc;")
        layout.addWidget(self.voice_status_label)
        
        # Voice transcription display
        self.voice_display = QTextEdit()
        self.voice_display.setMaximumHeight(80)
        self.voice_display.setReadOnly(True)
        self.voice_display.setStyleSheet("""
            QTextEdit {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }
        """)
        self.voice_display.setPlaceholderText("Voice transcriptions will appear here...")
        layout.addWidget(self.voice_display)
        
        return group_box
        
    def create_analysis_section(self):
        """Create AI analysis display section."""
        group_box = QGroupBox("🧠 AI Analysis & Results")
        group_box.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        layout = QVBoxLayout(group_box)
        
        # Analysis display
        self.analysis_display = QTextEdit()
        self.analysis_display.setReadOnly(True)
        self.analysis_display.setStyleSheet("""
            QTextEdit {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                line-height: 1.4;
            }
        """)
        self.analysis_display.setPlaceholderText("AI analysis results will appear here...")
        layout.addWidget(self.analysis_display)
        
        # Progress bar for ongoing operations
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                text-align: center;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        return group_box
        
    def create_history_section(self):
        """Create action history section."""
        group_box = QGroupBox("📋 Action History")
        group_box.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        layout = QVBoxLayout(group_box)
        
        # History display
        self.history_display = QTextEdit()
        self.history_display.setMaximumHeight(100)
        self.history_display.setReadOnly(True)
        self.history_display.setStyleSheet("""
            QTextEdit {
                background-color: #2a2a2a;
                color: #cccccc;
                border: 1px solid #555555;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 9px;
            }
        """)
        self.history_display.setPlaceholderText("Action history will appear here...")
        layout.addWidget(self.history_display)
        
        # Clear history button
        clear_btn = QPushButton("Clear History")
        clear_btn.setMaximumWidth(100)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: #ffffff;
                border: 1px solid #777777;
                border-radius: 3px;
                padding: 5px;
                font-size: 9px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        clear_btn.clicked.connect(self.clear_history)
        layout.addWidget(clear_btn, alignment=Qt.AlignRight)
        
        return group_box
    
    @pyqtSlot(str)
    def update_voice_status(self, status):
        """Update voice recognition status."""
        self.voice_status_label.setText(f"Status: {status}")
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
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #ffaa00;")
    
    def hide_processing(self):
        """Hide processing indicator."""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("color: #00ff00;")
    
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