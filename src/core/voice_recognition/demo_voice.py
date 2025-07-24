"""
Voice Handler Demo
=================

Demonstrates how to use the new voice handler with PyQt integration.
"""

import sys
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                           QWidget, QPushButton, QTextEdit, QLabel,
                           QHBoxLayout, QProgressBar)
from PyQt5.QtCore import QTimer

from voice_handler import VoiceHandler, test_microphone, list_audio_devices

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceDemo(QMainWindow):
    """Demo application for voice recognition."""
    
    def __init__(self):
        super().__init__()
        self.voice_handler = None
        self.init_ui()
        self.init_voice_handler()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Voice Recognition Demo")
        self.setGeometry(100, 100, 600, 400)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Status label
        self.status_label = QLabel("Voice Handler Status: Not initialized")
        layout.addWidget(self.status_label)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Listening")
        self.start_button.clicked.connect(self.start_listening)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Listening")
        self.stop_button.clicked.connect(self.stop_listening)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        self.test_mic_button = QPushButton("Test Microphone")
        self.test_mic_button.clicked.connect(self.test_microphone)
        button_layout.addWidget(self.test_mic_button)
        
        layout.addLayout(button_layout)
        
        # Recording indicator
        self.recording_label = QLabel("Recording: No")
        layout.addWidget(self.recording_label)
        
        # Progress bar (shows recording activity)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        # Transcription display
        self.transcription_display = QTextEdit()
        self.transcription_display.setPlaceholderText("Transcribed text will appear here...")
        self.transcription_display.setReadOnly(True)
        layout.addWidget(self.transcription_display)
        
        # Device info
        self.device_info = QTextEdit()
        self.device_info.setMaximumHeight(100)
        self.device_info.setPlaceholderText("Audio device information...")
        self.device_info.setReadOnly(True)
        layout.addWidget(self.device_info)
        
        # Load device info
        self.load_device_info()
    
    def init_voice_handler(self):
        """Initialize the voice handler."""
        try:
            self.voice_handler = VoiceHandler(self)
            
            # Connect signals
            self.voice_handler.transcription_received.connect(self.on_transcription)
            self.voice_handler.error_occurred.connect(self.on_error)
            self.voice_handler.recording_status_changed.connect(self.on_recording_status_changed)
            
            # Configuration
            config = {
                "audio": {
                    "sample_rate": 16000,
                    "silence_threshold": 0.01,
                    "silence_duration": 1.5,
                    "max_recording_duration": 30.0,
                    "min_recording_duration": 0.5
                },
                "whisper": {
                    "model_name": "base",
                    "device": "cpu",
                    "language": None,  # Auto-detect
                    "temperature": 0.0
                }
            }
            
            success = self.voice_handler.initialize(config)
            
            if success:
                self.status_label.setText("Voice Handler Status: Initialized successfully")
                self.start_button.setEnabled(True)
            else:
                self.status_label.setText("Voice Handler Status: Failed to initialize")
                
        except Exception as e:
            logger.error(f"Error initializing voice handler: {e}")
            self.status_label.setText(f"Voice Handler Status: Error - {e}")
    
    def start_listening(self):
        """Start voice recognition."""
        if self.voice_handler:
            success = self.voice_handler.start_listening()
            if success:
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.status_label.setText("Voice Handler Status: Listening...")
                self.transcription_display.append("=== Started listening ===\n")
            else:
                self.status_label.setText("Voice Handler Status: Failed to start listening")
    
    def stop_listening(self):
        """Stop voice recognition."""
        if self.voice_handler:
            self.voice_handler.stop_listening()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_label.setText("Voice Handler Status: Stopped")
            self.recording_label.setText("Recording: No")
            self.progress_bar.hide()
            self.transcription_display.append("=== Stopped listening ===\n")
    
    def test_microphone(self):
        """Test microphone functionality."""
        self.status_label.setText("Testing microphone...")
        self.test_mic_button.setEnabled(False)
        
        # Use QTimer to prevent UI blocking
        QTimer.singleShot(100, self._run_mic_test)
    
    def _run_mic_test(self):
        """Run the actual microphone test."""
        try:
            result = test_microphone()
            if result:
                self.status_label.setText("Microphone test: PASSED ✓")
                self.transcription_display.append("✓ Microphone test passed\n")
            else:
                self.status_label.setText("Microphone test: FAILED ✗")
                self.transcription_display.append("✗ Microphone test failed\n")
        except Exception as e:
            self.status_label.setText(f"Microphone test: ERROR - {e}")
            self.transcription_display.append(f"✗ Microphone test error: {e}\n")
        finally:
            self.test_mic_button.setEnabled(True)
    
    def load_device_info(self):
        """Load and display audio device information."""
        try:
            devices = list_audio_devices()
            device_text = "Available Audio Input Devices:\n"
            
            if devices:
                for device in devices:
                    device_text += f"  • {device['name']} (Index: {device['index']}, "
                    device_text += f"Channels: {device['channels']}, "
                    device_text += f"Sample Rate: {device['sample_rate']:.0f} Hz)\n"
            else:
                device_text += "  No input devices found"
            
            self.device_info.setText(device_text)
            
        except Exception as e:
            self.device_info.setText(f"Error loading device info: {e}")
    
    def on_transcription(self, text: str, confidence: float):
        """Handle transcription received."""
        timestamp = QTimer().currentTime().toString("hh:mm:ss")
        confidence_percent = confidence * 100
        
        formatted_text = f"[{timestamp}] (Confidence: {confidence_percent:.1f}%) {text}\n"
        self.transcription_display.append(formatted_text)
        
        # Auto-scroll to bottom
        scrollbar = self.transcription_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_error(self, error_message: str):
        """Handle voice handler errors."""
        self.status_label.setText(f"Voice Handler Status: Error - {error_message}")
        self.transcription_display.append(f"ERROR: {error_message}\n")
        
        # Reset button states
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.recording_label.setText("Recording: No")
        self.progress_bar.hide()
    
    def on_recording_status_changed(self, is_recording: bool):
        """Handle recording status changes."""
        if is_recording:
            self.recording_label.setText("Recording: Yes 🎙️")
            self.progress_bar.show()
        else:
            self.recording_label.setText("Recording: No")
            self.progress_bar.hide()
    
    def closeEvent(self, event):
        """Handle application close."""
        if self.voice_handler:
            self.voice_handler.cleanup()
        event.accept()

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Create and show the demo window
    demo = VoiceDemo()
    demo.show()
    
    # Run the application
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    main() 