"""
Voice Command Demo
=================

Demonstrates integration between voice recognition and command parsing.
This shows how to capture voice input, transcribe it with Whisper, and 
parse the commands for data analysis operations.

Usage:
    python voice_command_demo.py
"""

import sys
import json
import logging
from typing import Dict, Any

try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QTextEdit, QLabel
    from PyQt5.QtCore import pyqtSlot
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

from voice_handler import VoiceHandler
from command_parser import create_parser, CommandIntent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceCommandDemo(QMainWindow if PYQT_AVAILABLE else object):
    """Demo application for voice command recognition."""
    
    def __init__(self):
        if not PYQT_AVAILABLE:
            raise ImportError("PyQt5 is required for this demo")
        
        super().__init__()
        self.voice_handler = None
        self.command_parser = create_parser(use_nlp=False)
        self.setup_ui()
        self.setup_voice_handler()
    
    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("Voice Command Demo")
        self.setGeometry(100, 100, 600, 500)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("🎤 Voice Command Recognition Demo")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "Click 'Start Listening' and speak commands like:\n"
            "• 'visualize column age'\n"
            "• 'filter country equals Germany'\n"
            "• 'sort by score descending'\n"
            "• 'create bar chart for sales'"
        )
        instructions.setStyleSheet("margin: 10px; padding: 10px; background-color: #f0f0f0;")
        layout.addWidget(instructions)
        
        # Control button
        self.listen_button = QPushButton("Start Listening")
        self.listen_button.clicked.connect(self.toggle_listening)
        self.listen_button.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(self.listen_button)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("margin: 5px; padding: 5px; background-color: #e8f5e8;")
        layout.addWidget(self.status_label)
        
        # Results area
        results_label = QLabel("Transcription & Parsed Commands:")
        results_label.setStyleSheet("font-weight: bold; margin-top: 20px;")
        layout.addWidget(results_label)
        
        self.results_text = QTextEdit()
        self.results_text.setStyleSheet("font-family: monospace; font-size: 12px;")
        layout.addWidget(self.results_text)
        
        # Clear button
        clear_button = QPushButton("Clear Results")
        clear_button.clicked.connect(self.clear_results)
        layout.addWidget(clear_button)
    
    def setup_voice_handler(self):
        """Setup voice recognition handler."""
        try:
            self.voice_handler = VoiceHandler()
            
            # Configure for shorter recordings and quicker response
            config = {
                "audio": {
                    "silence_duration": 1.5,  # Stop after 1.5s of silence
                    "max_recording_duration": 10.0,  # Max 10 seconds
                    "silence_threshold": 0.01  # Sensitive silence detection
                },
                "whisper": {
                    "model_name": "base",
                    "temperature": 0.0
                }
            }
            
            if self.voice_handler.initialize(config):
                # Connect signals
                self.voice_handler.transcription_received.connect(self.on_transcription)
                self.voice_handler.error_occurred.connect(self.on_error)
                self.voice_handler.recording_status_changed.connect(self.on_recording_status)
                
                logger.info("Voice handler initialized successfully")
            else:
                logger.error("Failed to initialize voice handler")
                self.status_label.setText("❌ Voice handler initialization failed")
        
        except Exception as e:
            logger.error(f"Error setting up voice handler: {e}")
            self.status_label.setText(f"❌ Setup error: {e}")
    
    def toggle_listening(self):
        """Toggle voice listening on/off."""
        if not self.voice_handler:
            self.status_label.setText("❌ Voice handler not available")
            return
        
        if self.listen_button.text() == "Start Listening":
            if self.voice_handler.start_listening():
                self.listen_button.setText("Stop Listening")
                self.status_label.setText("🎤 Listening... Speak now!")
            else:
                self.status_label.setText("❌ Failed to start listening")
        else:
            self.voice_handler.stop_listening()
            self.listen_button.setText("Start Listening")
            self.status_label.setText("Ready")
    
    @pyqtSlot(str, float)
    def on_transcription(self, text: str, confidence: float):
        """Handle transcription results."""
        logger.info(f"Transcribed: {text} (confidence: {confidence:.2f})")
        
        # Parse the command
        parsed_command = self.command_parser.parse(text)
        
        # Display results
        self.display_result(text, confidence, parsed_command)
        
        # Auto-stop listening after getting transcription
        if self.listen_button.text() == "Stop Listening":
            self.voice_handler.stop_listening()
            self.listen_button.setText("Start Listening")
            self.status_label.setText("✅ Command processed")
    
    @pyqtSlot(str)
    def on_error(self, error_message: str):
        """Handle voice recognition errors."""
        logger.error(f"Voice error: {error_message}")
        self.status_label.setText(f"❌ Error: {error_message}")
        
        if self.listen_button.text() == "Stop Listening":
            self.listen_button.setText("Start Listening")
    
    @pyqtSlot(bool)
    def on_recording_status(self, is_recording: bool):
        """Handle recording status changes."""
        if is_recording:
            self.status_label.setText("🔴 Recording...")
        else:
            self.status_label.setText("⏹️ Processing...")
    
    def display_result(self, transcription: str, confidence: float, parsed_command):
        """Display transcription and parsing results."""
        result_text = "=" * 50 + "\n"
        result_text += f"🎤 TRANSCRIPTION: {transcription}\n"
        result_text += f"📊 CONFIDENCE: {confidence:.2f}\n"
        result_text += f"🤖 PARSED COMMAND:\n"
        
        command_dict = parsed_command.to_dict()
        result_text += json.dumps(command_dict, indent=2) + "\n"
        
        # Add interpretation
        result_text += f"💡 INTERPRETATION: "
        if parsed_command.intent == CommandIntent.VISUALIZE:
            if parsed_command.chart_type:
                result_text += f"Create {parsed_command.chart_type.value.replace('_', ' ')} for column '{parsed_command.column}'"
            else:
                result_text += f"Visualize column '{parsed_command.column}'"
        elif parsed_command.intent == CommandIntent.FILTER:
            result_text += f"Filter where {parsed_command.column} {parsed_command.operation.value} '{parsed_command.value}'"
        elif parsed_command.intent == CommandIntent.SORT:
            result_text += f"Sort by {parsed_command.column} in {parsed_command.sort_direction.value} order"
        elif parsed_command.intent == CommandIntent.AGGREGATE:
            result_text += f"Calculate {parsed_command.operation.value} of {parsed_command.column}"
        elif parsed_command.intent == CommandIntent.GROUP:
            result_text += f"Group data by {parsed_command.column}"
        elif parsed_command.intent == CommandIntent.SHOW:
            result_text += "Show/display data"
        elif parsed_command.intent == CommandIntent.DESCRIBE:
            result_text += "Describe/summarize data"
        else:
            result_text += "Unknown command"
        
        result_text += "\n\n"
        
        # Append to results
        self.results_text.append(result_text)
        
        # Scroll to bottom
        cursor = self.results_text.textCursor()
        cursor.movePosition(cursor.End)
        self.results_text.setTextCursor(cursor)
    
    def clear_results(self):
        """Clear the results area."""
        self.results_text.clear()
    
    def closeEvent(self, event):
        """Handle application close."""
        if self.voice_handler:
            self.voice_handler.cleanup()
        event.accept()

def run_command_line_demo():
    """Run a simple command-line demo."""
    print("🎤 Voice Command Parser Demo (Command Line)")
    print("=" * 50)
    print("Enter text commands to see how they're parsed:")
    print("(or 'quit' to exit)")
    print()
    
    parser = create_parser(use_nlp=False)
    
    test_commands = [
        "visualize column age",
        "filter country equals Germany",
        "sort by score descending",
        "create bar chart for sales",
        "sum of revenue"
    ]
    
    print("Example commands:")
    for cmd in test_commands:
        print(f"  • {cmd}")
    print()
    
    while True:
        try:
            user_input = input("Command: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Parse the command
            result = parser.parse(user_input)
            
            print(f"Parsed: {json.dumps(result.to_dict(), indent=2)}")
            print()
            
        except KeyboardInterrupt:
            break
    
    print("Goodbye! 👋")

def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_command_line_demo()
        return
    
    if not PYQT_AVAILABLE:
        print("PyQt5 not available. Running command-line demo instead.")
        print("Install PyQt5 for the GUI version: pip install PyQt5")
        print()
        run_command_line_demo()
        return
    
    app = QApplication(sys.argv)
    demo = VoiceCommandDemo()
    demo.show()
    
    print("🎤 Voice Command Demo started!")
    print("Use the GUI to test voice recognition and command parsing.")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 