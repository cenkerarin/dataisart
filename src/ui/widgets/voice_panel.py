"""
Voice Panel Widget
==================

Voice recognition panel that integrates with gesture detection.
Activates voice recognition when the "peace_sign" gesture is detected,
transcribes speech using Whisper, and parses commands for data analysis.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QGroupBox, QTextEdit,
                             QProgressBar, QCheckBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor

# Add voice recognition path
voice_recognition_path = Path(__file__).parent.parent.parent / "core" / "voice_recognition"
sys.path.insert(0, str(voice_recognition_path))

try:
    from voice_handler import VoiceHandler, test_microphone, list_audio_devices
    from command_parser import create_parser, CommandIntent, ParsedCommand
    VOICE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Voice recognition not available: {e}")
    VOICE_AVAILABLE = False

logger = logging.getLogger(__name__)

class VoicePanel(QWidget):
    """Voice recognition panel with gesture-based activation."""
    
    # Signals
    command_parsed = pyqtSignal(dict)  # Emitted when a command is successfully parsed
    voice_status_changed = pyqtSignal(str)  # Status updates
    gesture_trigger_detected = pyqtSignal(str)  # When trigger gesture is detected
    transcription_received = pyqtSignal(str)  # Emitted when transcription is received
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Debug: Show voice availability
        print(f"🔊 Voice Panel - VOICE_AVAILABLE: {VOICE_AVAILABLE}")
        
        # Voice components
        self.voice_handler = None
        self.command_parser = None
        self.is_voice_active = False
        self.is_listening = False
        
        # Gesture activation settings
        self.trigger_gesture = "peace_sign"
        self.gesture_hold_duration = 1.0  # seconds to hold gesture
        self.gesture_cooldown = 3.0  # seconds before next activation
        
        # State tracking
        self.gesture_start_time = None
        self.last_activation_time = 0
        self.current_gesture = None
        self.manually_stopped = False  # Track manual stops due to gesture removal
        
        # UI setup
        self.setup_ui()
        self.setup_voice_components()
        
        # Gesture monitoring timer
        self.gesture_timer = QTimer()
        self.gesture_timer.timeout.connect(self.check_gesture_activation)
        self.gesture_timer.start(100)  # Check every 100ms
    
    def setup_ui(self):
        """Setup the voice panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("🎤 Voice Command Center")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                padding: 10px;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                background-color: rgba(76, 175, 80, 0.1);
            }
        """)
        layout.addWidget(title)
        
        # Status section
        status_group = QGroupBox("Status")
        status_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        status_layout = QVBoxLayout(status_group)
        
        # Voice status
        self.voice_status_label = QLabel("🔇 Voice Recognition: Inactive")
        self.voice_status_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        status_layout.addWidget(self.voice_status_label)
        
        # Gesture status
        self.gesture_status_label = QLabel("✋ Gesture: None detected")
        self.gesture_status_label.setStyleSheet("color: #95a5a6;")
        status_layout.addWidget(self.gesture_status_label)
        
        # Activation progress
        self.activation_progress = QProgressBar()
        self.activation_progress.setVisible(False)
        self.activation_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4CAF50;
                border-radius: 5px;
                background-color: #2b2b2b;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        status_layout.addWidget(self.activation_progress)
        
        layout.addWidget(status_group)
        
        # Settings section
        settings_group = QGroupBox("Settings")
        settings_group.setStyleSheet(status_group.styleSheet())
        settings_layout = QVBoxLayout(settings_group)
        
        # Enable/disable voice recognition
        self.voice_enabled_checkbox = QCheckBox("Enable Voice Recognition")
        self.voice_enabled_checkbox.setChecked(True)
        self.voice_enabled_checkbox.stateChanged.connect(self.toggle_voice_recognition)
        settings_layout.addWidget(self.voice_enabled_checkbox)
        
        # Trigger gesture info
        trigger_info = QLabel(f"✌️ Trigger Gesture: {self.trigger_gesture.replace('_', ' ').title()}")
        trigger_info.setStyleSheet("color: #74b9ff; font-style: italic;")
        settings_layout.addWidget(trigger_info)
        
        # Hold duration info
        duration_info = QLabel(f"⏱️ Hold Duration: {self.gesture_hold_duration}s")
        duration_info.setStyleSheet("color: #74b9ff; font-style: italic;")
        settings_layout.addWidget(duration_info)
        
        layout.addWidget(settings_group)
        
        # Manual controls
        controls_group = QGroupBox("Manual Controls")
        controls_group.setStyleSheet(status_group.styleSheet())
        controls_layout = QVBoxLayout(controls_group)
        
        # Manual activation button
        self.manual_button = QPushButton("🎤 Start Voice Recognition")
        self.manual_button.clicked.connect(self.toggle_manual_listening)
        self.manual_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        controls_layout.addWidget(self.manual_button)
        
        # Test microphone button
        test_mic_button = QPushButton("🔊 Test Microphone")
        test_mic_button.clicked.connect(self.test_microphone)
        test_mic_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        controls_layout.addWidget(test_mic_button)
        
        layout.addWidget(controls_group)
        
        # Command history
        history_group = QGroupBox("Command History")
        history_group.setStyleSheet(status_group.styleSheet())
        history_layout = QVBoxLayout(history_group)
        
        self.command_history = QTextEdit()
        self.command_history.setMaximumHeight(150)
        self.command_history.setReadOnly(True)
        self.command_history.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }
        """)
        history_layout.addWidget(self.command_history)
        
        # Clear history button
        clear_button = QPushButton("Clear History")
        clear_button.clicked.connect(self.command_history.clear)
        clear_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        history_layout.addWidget(clear_button)
        
        layout.addWidget(history_group)
        
        # Instructions
        instructions = QLabel(
            "Instructions:\n"
            "1. Show ✌️ peace sign to camera\n"
            "2. Hold for 1 second to activate\n"
            "3. Speak your command clearly\n"
            "4. Wait for processing"
        )
        instructions.setStyleSheet("""
            QLabel {
                color: #95a5a6;
                font-size: 11px;
                font-style: italic;
                padding: 10px;
                border: 1px solid #555;
                border-radius: 5px;
                background-color: rgba(149, 165, 166, 0.1);
            }
        """)
        layout.addWidget(instructions)
        
        layout.addStretch()
    
    def setup_voice_components(self):
        """Initialize voice recognition components."""
        if not VOICE_AVAILABLE:
            self.voice_status_label.setText("❌ Voice components not available")
            self.voice_enabled_checkbox.setEnabled(False)
            self.manual_button.setEnabled(False)
            return
        
        try:
            # Initialize voice handler
            self.voice_handler = VoiceHandler()
            
            # Configure for quick response
            config = {
                "audio": {
                    "silence_duration": 1.5,  # Stop after 1.5s of silence
                    "max_recording_duration": 10.0,  # Max 10 seconds
                    "silence_threshold": 0.01,  # Sensitive silence detection
                    "min_recording_duration": 0.5  # Minimum 0.5s recording
                },
                "whisper": {
                    "model_name": "base",
                    "temperature": 0.0
                }
            }
            
            if self.voice_handler.initialize(config):
                # Connect signals
                self.voice_handler.transcription_received.connect(self.on_transcription_received)
                self.voice_handler.error_occurred.connect(self.on_voice_error)
                self.voice_handler.recording_status_changed.connect(self.on_recording_status_changed)
                
                # Initialize command parser
                self.command_parser = create_parser(use_nlp=False)
                
                self.voice_status_label.setText("✅ Voice Recognition: Ready")
                self.voice_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                
                logger.info("Voice components initialized successfully")
            else:
                raise Exception("Voice handler initialization failed")
                
        except Exception as e:
            logger.error(f"Failed to initialize voice components: {e}")
            self.voice_status_label.setText(f"❌ Voice Error: {str(e)}")
            self.voice_enabled_checkbox.setEnabled(False)
            self.manual_button.setEnabled(False)
    
    def update_gesture_status(self, gesture: str, confidence: float = 0.0):
        """Update current gesture status from camera panel."""
        # Debug: Print all gestures for troubleshooting
        print(f"🔍 Voice Panel - Gesture received: {gesture} (confidence: {confidence:.2f})")
        
        # Set current_gesture to None if confidence is too low or gesture is "none"
        self.current_gesture = gesture if (confidence > 0.5 and gesture != "none") else None  # Lowered threshold
        
        if self.current_gesture:
            self.gesture_status_label.setText(f"✋ Gesture: {gesture.replace('_', ' ').title()} ({confidence:.2f})")
            self.gesture_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
            # Debug: Check if it's the trigger gesture
            if gesture == self.trigger_gesture:
                print(f"🎯 Voice Panel - Trigger gesture detected! ({self.trigger_gesture})")
        else:
            self.gesture_status_label.setText("✋ Gesture: None detected")
            self.gesture_status_label.setStyleSheet("color: #95a5a6;")
    
    def check_gesture_activation(self):
        """Check if gesture activation should trigger voice recognition."""
        if not self.voice_enabled_checkbox.isChecked() or not VOICE_AVAILABLE:
            return
        
        current_time = time.time()
        
        # Check if we're in cooldown period
        if current_time - self.last_activation_time < self.gesture_cooldown:
            return
        
        # Check if trigger gesture is detected
        if self.current_gesture == self.trigger_gesture:
            print(f"🎯 Voice Panel - Activating voice for gesture: {self.current_gesture}")
            if self.gesture_start_time is None:
                self.gesture_start_time = current_time
                self.activation_progress.setVisible(True)
                self.activation_progress.setValue(0)
            else:
                # Calculate hold duration
                hold_duration = current_time - self.gesture_start_time
                progress = min(100, int((hold_duration / self.gesture_hold_duration) * 100))
                self.activation_progress.setValue(progress)
                
                # Activate if held long enough
                if hold_duration >= self.gesture_hold_duration:
                    self.activate_voice_recognition()
                    self.gesture_start_time = None
                    self.activation_progress.setVisible(False)
                    self.last_activation_time = current_time
        else:
            # Reset if gesture is lost
            if self.gesture_start_time is not None:
                self.gesture_start_time = None
                self.activation_progress.setVisible(False)
            
            # Stop voice recognition if currently listening and gesture is lost
            if self.is_listening and self.voice_handler:
                self.manually_stopped = True  # Flag that we manually stopped
                # Temporarily disconnect recording status signal to prevent race condition
                self.voice_handler.recording_status_changed.disconnect()
                self.voice_handler.stop_listening()
                # Reconnect the signal
                self.voice_handler.recording_status_changed.connect(self.on_recording_status_changed)
                self.is_listening = False
                self.voice_status_label.setText("✅ Voice Recognition: Ready")
                self.voice_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                self.add_to_history("🛑 Voice recognition stopped - gesture removed")
    
    def activate_voice_recognition(self):
        """Activate voice recognition through gesture trigger."""
        print(f"🚀 Voice Panel - activate_voice_recognition called! is_listening: {self.is_listening}")
        
        if self.is_listening:
            return
        
        self.gesture_trigger_detected.emit(self.trigger_gesture)
        self.add_to_history("🎯 Gesture activation detected!")
        
        print(f"🔊 Voice Panel - voice_handler available: {self.voice_handler is not None}")
        
        if self.voice_handler and self.voice_handler.start_listening():
            self.is_listening = True
            self.manually_stopped = False  # Reset manual stop flag
            self.voice_status_label.setText("🎤 Listening...")
            self.voice_status_label.setStyleSheet("color: #ff6b6b; font-weight: bold; animation: blink 1s infinite;")
            logger.info("Voice recognition activated by gesture")
            print("✅ Voice Panel - Voice recognition started successfully!")
        else:
            self.add_to_history("❌ Failed to start voice recognition")
            print("❌ Voice Panel - Failed to start voice recognition")
    
    def toggle_manual_listening(self):
        """Toggle manual voice recognition."""
        if not VOICE_AVAILABLE or not self.voice_handler:
            return
        
        if self.is_listening:
            self.manually_stopped = True  # Flag manual stop
            # Temporarily disconnect recording status signal to prevent race condition
            self.voice_handler.recording_status_changed.disconnect()
            self.voice_handler.stop_listening()
            # Reconnect the signal
            self.voice_handler.recording_status_changed.connect(self.on_recording_status_changed)
            self.is_listening = False
            self.manual_button.setText("🎤 Start Voice Recognition")
            self.voice_status_label.setText("✅ Voice Recognition: Ready")
            self.voice_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            if self.voice_handler.start_listening():
                self.is_listening = True
                self.manually_stopped = False  # Reset manual stop flag
                self.manual_button.setText("🛑 Stop Listening")
                self.voice_status_label.setText("🎤 Listening...")
                self.voice_status_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
                self.add_to_history("🎤 Manual voice recognition started")
    
    def toggle_voice_recognition(self, state):
        """Enable/disable voice recognition."""
        self.is_voice_active = state == Qt.Checked
        if not self.is_voice_active and self.is_listening:
            self.voice_handler.stop_listening()
            self.is_listening = False
            self.manual_button.setText("🎤 Start Voice Recognition")
    
    def test_microphone(self):
        """Test microphone functionality."""
        if not VOICE_AVAILABLE:
            self.add_to_history("❌ Voice components not available")
            return
        
        self.add_to_history("🔊 Testing microphone...")
        
        try:
            if test_microphone():
                self.add_to_history("✅ Microphone test successful")
            else:
                self.add_to_history("❌ Microphone test failed - no audio detected")
        except Exception as e:
            self.add_to_history(f"❌ Microphone test error: {str(e)}")
    
    @pyqtSlot(str, float)
    def on_transcription_received(self, text: str, confidence: float):
        """Handle transcription from voice recognition."""
        self.add_to_history(f"🎤 Transcribed: {text} (confidence: {confidence:.2f})")
        
        # Emit transcription for AI action panel
        self.transcription_received.emit(f"{text} (confidence: {confidence:.2f})")
        
        # Parse the command
        if self.command_parser:
            parsed_command = self.command_parser.parse(text)
            self.add_to_history(f"🤖 Intent: {parsed_command.intent.value}")
            
            if parsed_command.intent != CommandIntent.UNKNOWN:
                # Add command details
                details = []
                if parsed_command.column:
                    details.append(f"column: {parsed_command.column}")
                if parsed_command.operation:
                    details.append(f"operation: {parsed_command.operation.value}")
                if parsed_command.value:
                    details.append(f"value: {parsed_command.value}")
                if parsed_command.chart_type:
                    details.append(f"chart: {parsed_command.chart_type.value}")
                if parsed_command.sort_direction:
                    details.append(f"direction: {parsed_command.sort_direction.value}")
                
                if details:
                    self.add_to_history(f"   └─ {', '.join(details)}")
                
                # Emit parsed command
                self.command_parsed.emit(parsed_command.to_dict())
            else:
                self.add_to_history("   └─ ⚠️ Command not recognized")
        
        # Reset listening state
        self.is_listening = False
        self.manually_stopped = False  # Reset manual stop flag on successful transcription
        self.manual_button.setText("🎤 Start Voice Recognition")
        self.voice_status_label.setText("✅ Voice Recognition: Ready")
        self.voice_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
    
    @pyqtSlot(str)
    def on_voice_error(self, error_message: str):
        """Handle voice recognition errors."""
        self.add_to_history(f"❌ Voice Error: {error_message}")
        
        # Reset listening state
        self.is_listening = False
        self.manually_stopped = False  # Reset manual stop flag on error
        self.manual_button.setText("🎤 Start Voice Recognition")
        self.voice_status_label.setText("✅ Voice Recognition: Ready")
        self.voice_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
    
    @pyqtSlot(bool)
    def on_recording_status_changed(self, is_recording: bool):
        """Handle recording status changes."""
        if is_recording:
            self.manually_stopped = False  # Reset manual stop flag when recording starts
            self.voice_status_label.setText("🔴 Recording...")
            self.voice_status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        else:
            # Only show "Processing" if we didn't manually stop
            if not self.manually_stopped:
                self.voice_status_label.setText("⏳ Processing...")
                self.voice_status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
            # If manually stopped, the status was already set to "Ready"
    
    def add_to_history(self, message: str):
        """Add message to command history."""
        import time
        timestamp = time.strftime("%H:%M:%S")
        self.command_history.append(f"[{timestamp}] {message}")
        
        # Auto-scroll to bottom
        cursor = self.command_history.textCursor()
        cursor.movePosition(cursor.End)
        self.command_history.setTextCursor(cursor)
    
    def cleanup(self):
        """Clean up voice components."""
        if self.voice_handler:
            self.voice_handler.cleanup()
        
        if hasattr(self, 'gesture_timer'):
            self.gesture_timer.stop()

# For testing the voice panel standalone
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyleSheet("""
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
    """)
    
    voice_panel = VoicePanel()
    voice_panel.show()
    
    # Simulate gesture updates for testing
    def simulate_gestures():
        import random
        gestures = ["neutral", "pointing", "peace_sign", "fist", "open_hand"]
        gesture = random.choice(gestures)
        confidence = random.uniform(0.5, 0.95)
        voice_panel.update_gesture_status(gesture, confidence)
    
    # Update gestures every 2 seconds for demo
    gesture_timer = QTimer()
    gesture_timer.timeout.connect(simulate_gestures)
    gesture_timer.start(2000)
    
    sys.exit(app.exec_()) 