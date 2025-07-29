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
        """Setup the unified AI chat interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # Chat header with title and controls
        header_layout = QHBoxLayout()
        
        title_label = QLabel("🤖 AI Chat")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setStyleSheet("color: #ffffff;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Voice status indicator (compact)
        self.voice_status_label = QLabel("🎤 Voice: Inactive")
        self.voice_status_label.setStyleSheet("color: #888888; font-size: 11px;")
        header_layout.addWidget(self.voice_status_label)
        
        # Clear chat button
        clear_btn = QPushButton("Clear")
        clear_btn.setMaximumWidth(50)
        clear_btn.setMaximumHeight(24)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #444444;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
        """)
        clear_btn.clicked.connect(self.clear_chat)
        header_layout.addWidget(clear_btn)
        
        main_layout.addLayout(header_layout)
        
        # Progress bar for ongoing operations (initially hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(3)
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
        main_layout.addWidget(self.progress_bar)
        
        # Main chat area - unified conversation interface
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 6px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
                line-height: 1.4;
                padding: 12px;
            }
        """)
        self.chat_display.setPlaceholderText("🤖 AI Chat ready...\n\nVoice transcriptions, AI analysis, and actions will appear here as a conversation.")
        main_layout.addWidget(self.chat_display)

    def add_message(self, message_type, content, metadata=None):
        """Add a message to the unified chat interface."""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Format message based on type
        if message_type == "voice":
            icon = "🎤"
            title = "Voice"
            color = "#00ff88"
            formatted_content = content
        elif message_type == "ai":
            icon = "🧠"
            title = "AI Analysis"
            color = "#0078d4"
            formatted_content = content
        elif message_type == "action":
            icon = "⚡"
            title = "Action"
            color = "#ffaa00"
            formatted_content = content
        elif message_type == "system":
            icon = "ℹ️"
            title = "System"
            color = "#888888"
            formatted_content = content
        elif message_type == "error":
            icon = "❌"
            title = "Error"
            color = "#ff4444"
            formatted_content = content
        else:
            icon = "💬"
            title = "Message"
            color = "#cccccc"
            formatted_content = content
        
        # Create message HTML with chat-like styling
        message_html = f"""
        <div style="margin-bottom: 16px; border-left: 3px solid {color}; padding-left: 12px;">
            <div style="color: {color}; font-weight: bold; font-size: 11px; margin-bottom: 4px;">
                {icon} {title} <span style="color: #666666; font-weight: normal;">• {timestamp}</span>
            </div>
            <div style="color: #ffffff; line-height: 1.4;">
                {formatted_content}
            </div>
        </div>
        """
        
        # Append message to chat
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertHtml(message_html)
        
        # Auto-scroll to bottom
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def clear_chat(self):
        """Clear the chat interface."""
        self.chat_display.clear()
        self.action_history.clear()
        self.add_message("system", "Chat cleared")

    # Update methods to use the unified chat interface
    @pyqtSlot(str)
    def update_voice_status(self, status):
        """Update voice recognition status."""
        self.voice_status_label.setText(f"🎤 Voice: {status}")
        if "listening" in status.lower():
            self.voice_status_label.setStyleSheet("color: #00ff88; font-size: 11px;")
        elif "processing" in status.lower():
            self.voice_status_label.setStyleSheet("color: #ffaa00; font-size: 11px;")
        else:
            self.voice_status_label.setStyleSheet("color: #888888; font-size: 11px;")
    
    @pyqtSlot(str)
    def update_voice_transcription(self, transcription):
        """Update voice transcription in chat."""
        self.add_message("voice", transcription)
    
    @pyqtSlot(dict)
    def handle_voice_command(self, command_data):
        """Handle parsed voice command in chat."""
        command = command_data.get('command', 'Unknown')
        self.add_message("action", f"Voice Command: {command}")
        
        # Show processing
        self.show_processing("Processing voice command...")
        
        # If AI assistant is available, process the command
        if self.ai_assistant and LLM_AVAILABLE:
            try:
                result = self.ai_assistant.process_command(command_data)
                self.display_ai_result(result)
            except Exception as e:
                self.add_message("error", f"Error processing command: {str(e)}")
        else:
            self.add_message("system", "Voice command received (AI processing not available)")
        
        self.hide_processing()
    
    def display_ai_result(self, result):
        """Display AI analysis result in chat."""
        if not result:
            return
            
        # Format result based on type
        if hasattr(result, 'action_type'):
            content = f"**Action:** {result.action_type}\n"
            
            if hasattr(result, 'result') and result.result:
                content += f"**Result:** {result.result}\n"
                
            if hasattr(result, 'explanation') and result.explanation:
                content += f"**Explanation:** {result.explanation}\n"
                
            if hasattr(result, 'confidence') and result.confidence:
                content += f"**Confidence:** {result.confidence:.2f}"
        else:
            content = str(result)
        
        self.add_message("ai", content)
    
    def display_message(self, message):
        """Display a general message in chat."""
        self.add_message("system", message)
    
    def display_error(self, error_message):
        """Display error message in chat."""
        self.add_message("error", error_message)
    
    def add_to_history(self, action):
        """Add action to chat (replaces old history system)."""
        self.add_message("action", action)

    def clear_history(self):
        """Legacy method - now calls clear_chat"""
        self.clear_chat()

    # Remove the old section setup methods
    def create_voice_section(self):
        """Legacy method - now integrated into chat"""
        pass
        
    def create_analysis_section(self):
        """Legacy method - now integrated into chat"""
        pass
        
    def create_history_section(self):
        """Legacy method - now integrated into chat"""
        pass
    
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