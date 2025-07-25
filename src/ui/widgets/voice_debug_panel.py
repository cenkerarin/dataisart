"""
Voice Debug Panel Widget
========================

Enhanced voice recognition panel with detailed debugging information
to help troubleshoot voice command recognition issues.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QGroupBox, QTextEdit,
                             QProgressBar, QCheckBox, QSlider, QLCDNumber)
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

class VoiceDebugPanel(QWidget):
    """Voice recognition panel with extensive debugging information."""
    
    # Signals
    command_parsed = pyqtSignal(dict)  # Emitted when a command is successfully parsed
    voice_status_changed = pyqtSignal(str)  # Status updates
    gesture_trigger_detected = pyqtSignal(str)  # When trigger gesture is detected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
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
        
        # Debug counters
        self.total_activations = 0
        self.successful_transcriptions = 0
        self.successful_parses = 0
        self.last_transcription_time = None
        
        # UI setup
        self.setup_ui()
        self.setup_voice_components()
        
        # Gesture monitoring timer
        self.gesture_timer = QTimer()
        self.gesture_timer.timeout.connect(self.check_gesture_activation)
        self.gesture_timer.start(100)  # Check every 100ms
        
        # Audio level monitoring timer
        self.audio_timer = QTimer()
        self.audio_timer.timeout.connect(self.update_audio_levels)
        self.audio_timer.start(50)  # Update every 50ms for smooth display
    
    def setup_ui(self):
        """Setup the debug voice panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("🎤 Voice Debug Center")
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
        
        # Audio levels section
        self.setup_audio_levels_section(layout)
        
        # Status section
        self.setup_status_section(layout)
        
        # Debug statistics
        self.setup_debug_stats_section(layout)
        
        # Manual controls
        self.setup_controls_section(layout)
        
        # Detailed logs
        self.setup_logs_section(layout)
        
        # Instructions
        self.setup_instructions_section(layout)
        
        layout.addStretch()
    
    def setup_audio_levels_section(self, layout):
        """Setup audio level monitoring section."""
        audio_group = QGroupBox("🔊 Audio Monitoring")
        audio_group.setStyleSheet("""
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
        audio_layout = QVBoxLayout(audio_group)
        
        # Audio level bar
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel("Level:"))
        
        self.audio_level_bar = QProgressBar()
        self.audio_level_bar.setRange(0, 100)
        self.audio_level_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                background-color: #2b2b2b;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:0.7 #FFC107, stop:1 #F44336);
                border-radius: 3px;
            }
        """)
        level_layout.addWidget(self.audio_level_bar)
        
        self.audio_level_value = QLabel("0.000")
        self.audio_level_value.setMinimumWidth(50)
        self.audio_level_value.setStyleSheet("font-family: monospace; color: #4CAF50;")
        level_layout.addWidget(self.audio_level_value)
        
        audio_layout.addLayout(level_layout)
        
        # Silence threshold indicator
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Silence Threshold:"))
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 100)  # 0.001 to 0.1
        self.threshold_slider.setValue(20)  # 0.02 default
        self.threshold_slider.valueChanged.connect(self.update_silence_threshold)
        threshold_layout.addWidget(self.threshold_slider)
        
        self.threshold_value = QLabel("0.020")
        self.threshold_value.setMinimumWidth(50)
        self.threshold_value.setStyleSheet("font-family: monospace; color: #FFC107;")
        threshold_layout.addWidget(self.threshold_value)
        
        audio_layout.addLayout(threshold_layout)
        
        # Recording timer
        timer_layout = QHBoxLayout()
        timer_layout.addWidget(QLabel("Recording Time:"))
        
        self.recording_timer = QLCDNumber(4)
        self.recording_timer.setStyleSheet("""
            QLCDNumber {
                background-color: #1e1e1e;
                color: #4CAF50;
                border: 1px solid #555;
                border-radius: 4px;
            }
        """)
        self.recording_timer.display("0.00")
        timer_layout.addWidget(self.recording_timer)
        
        audio_layout.addLayout(timer_layout)
        
        layout.addWidget(audio_group)
    
    def setup_status_section(self, layout):
        """Setup status monitoring section."""
        status_group = QGroupBox("📊 Status")
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
        
        # Last transcription
        self.transcription_status_label = QLabel("💬 Last Transcription: None")
        self.transcription_status_label.setStyleSheet("color: #74b9ff;")
        status_layout.addWidget(self.transcription_status_label)
        
        # Command parsing status
        self.parsing_status_label = QLabel("🤖 Last Parse: None")
        self.parsing_status_label.setStyleSheet("color: #a29bfe;")
        status_layout.addWidget(self.parsing_status_label)
        
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
    
    def setup_debug_stats_section(self, layout):
        """Setup debug statistics section."""
        stats_group = QGroupBox("📈 Debug Statistics")
        stats_group.setStyleSheet("""
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
        stats_layout = QVBoxLayout(stats_group)
        
        # Statistics labels
        self.total_activations_label = QLabel("🎯 Total Activations: 0")
        self.total_activations_label.setStyleSheet("color: #74b9ff;")
        stats_layout.addWidget(self.total_activations_label)
        
        self.successful_transcriptions_label = QLabel("✅ Successful Transcriptions: 0")
        self.successful_transcriptions_label.setStyleSheet("color: #00b894;")
        stats_layout.addWidget(self.successful_transcriptions_label)
        
        self.successful_parses_label = QLabel("🤖 Successful Parses: 0")
        self.successful_parses_label.setStyleSheet("color: #a29bfe;")
        stats_layout.addWidget(self.successful_parses_label)
        
        # Success rates
        self.transcription_rate_label = QLabel("📊 Transcription Rate: 0%")
        self.transcription_rate_label.setStyleSheet("color: #fdcb6e;")
        stats_layout.addWidget(self.transcription_rate_label)
        
        self.parsing_rate_label = QLabel("🎯 Parsing Rate: 0%")
        self.parsing_rate_label.setStyleSheet("color: #e17055;")
        stats_layout.addWidget(self.parsing_rate_label)
        
        # Reset button
        reset_stats_button = QPushButton("🔄 Reset Statistics")
        reset_stats_button.clicked.connect(self.reset_statistics)
        reset_stats_button.setStyleSheet("""
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
        stats_layout.addWidget(reset_stats_button)
        
        layout.addWidget(stats_group)
    
    def setup_controls_section(self, layout):
        """Setup manual controls section."""
        controls_group = QGroupBox("🎮 Manual Controls")
        controls_group.setStyleSheet("""
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
        controls_layout = QVBoxLayout(controls_group)
        
        # Enable/disable voice recognition
        self.voice_enabled_checkbox = QCheckBox("Enable Voice Recognition")
        self.voice_enabled_checkbox.setChecked(True)
        self.voice_enabled_checkbox.stateChanged.connect(self.toggle_voice_recognition)
        controls_layout.addWidget(self.voice_enabled_checkbox)
        
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
        
        # Test buttons row
        test_layout = QHBoxLayout()
        
        # Test microphone button
        test_mic_button = QPushButton("🔊 Test Mic")
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
        test_layout.addWidget(test_mic_button)
        
        # Test command parsing button
        test_parse_button = QPushButton("🤖 Test Parse")
        test_parse_button.clicked.connect(self.test_command_parsing)
        test_parse_button.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        test_layout.addWidget(test_parse_button)
        
        controls_layout.addLayout(test_layout)
        
        layout.addWidget(controls_group)
    
    def setup_logs_section(self, layout):
        """Setup detailed logs section."""
        logs_group = QGroupBox("📋 Detailed Logs")
        logs_group.setStyleSheet("""
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
        logs_layout = QVBoxLayout(logs_group)
        
        self.debug_logs = QTextEdit()
        self.debug_logs.setMaximumHeight(200)
        self.debug_logs.setReadOnly(True)
        self.debug_logs.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }
        """)
        logs_layout.addWidget(self.debug_logs)
        
        # Log controls
        log_controls_layout = QHBoxLayout()
        
        clear_logs_button = QPushButton("Clear Logs")
        clear_logs_button.clicked.connect(self.debug_logs.clear)
        clear_logs_button.setStyleSheet("""
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
        log_controls_layout.addWidget(clear_logs_button)
        
        # Auto-scroll checkbox
        self.auto_scroll_checkbox = QCheckBox("Auto-scroll")
        self.auto_scroll_checkbox.setChecked(True)
        log_controls_layout.addWidget(self.auto_scroll_checkbox)
        
        log_controls_layout.addStretch()
        logs_layout.addLayout(log_controls_layout)
        
        layout.addWidget(logs_group)
    
    def setup_instructions_section(self, layout):
        """Setup instructions section."""
        instructions = QLabel(
            "🔧 Debug Instructions:\n"
            "1. Start camera and enable voice recognition\n"
            "2. Watch audio levels when speaking\n"
            "3. Show ✌️ peace sign for 1 second OR use manual button\n"
            "4. Speak clearly: 'describe data', 'filter age greater than 25'\n"
            "5. Check logs for detailed transcription and parsing info\n"
            "6. Adjust silence threshold if needed"
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
    
    def setup_voice_components(self):
        """Initialize voice recognition components with debug logging."""
        if not VOICE_AVAILABLE:
            self.voice_status_label.setText("❌ Voice components not available")
            self.voice_enabled_checkbox.setEnabled(False)
            self.manual_button.setEnabled(False)
            self.add_debug_log("❌ Voice components not available - check imports")
            return
        
        try:
            # Initialize voice handler
            self.voice_handler = VoiceHandler()
            
            # Configure for debug mode
            config = {
                "audio": {
                    "silence_duration": 1.5,
                    "max_recording_duration": 10.0,
                    "silence_threshold": 0.02,
                    "min_recording_duration": 0.5
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
                
                self.add_debug_log("✅ Voice components initialized successfully")
                self.add_debug_log(f"🎤 Audio config: silence_threshold={config['audio']['silence_threshold']}")
                self.add_debug_log(f"🤖 Whisper model: {config['whisper']['model_name']}")
                
                logger.info("Voice components initialized successfully")
            else:
                raise Exception("Voice handler initialization failed")
                
        except Exception as e:
            logger.error(f"Failed to initialize voice components: {e}")
            self.voice_status_label.setText(f"❌ Voice Error: {str(e)}")
            self.voice_enabled_checkbox.setEnabled(False)
            self.manual_button.setEnabled(False)
            self.add_debug_log(f"❌ Voice initialization error: {str(e)}")
    
    def update_audio_levels(self):
        """Update audio level display (simulated for now)."""
        # This would need integration with actual audio input monitoring
        # For now, show simulated levels during recording
        if self.is_listening:
            import random
            level = random.uniform(0.1, 0.8) if random.random() > 0.3 else random.uniform(0.0, 0.1)
            self.audio_level_bar.setValue(int(level * 100))
            self.audio_level_value.setText(f"{level:.3f}")
            
            # Update recording timer
            if hasattr(self, 'recording_start_time') and self.recording_start_time:
                elapsed = time.time() - self.recording_start_time
                self.recording_timer.display(f"{elapsed:.2f}")
        else:
            self.audio_level_bar.setValue(0)
            self.audio_level_value.setText("0.000")
            self.recording_timer.display("0.00")
    
    def update_silence_threshold(self, value):
        """Update silence threshold setting."""
        threshold = value / 1000.0  # Convert to 0.001 - 0.1 range
        self.threshold_value.setText(f"{threshold:.3f}")
        
        # Update voice handler configuration if available
        if self.voice_handler and hasattr(self.voice_handler, 'thread'):
            self.voice_handler.thread.audio_config.silence_threshold = threshold
            self.add_debug_log(f"🎚️ Silence threshold updated to {threshold:.3f}")
    
    def reset_statistics(self):
        """Reset debug statistics."""
        self.total_activations = 0
        self.successful_transcriptions = 0
        self.successful_parses = 0
        self.update_statistics_display()
        self.add_debug_log("🔄 Statistics reset")
    
    def update_statistics_display(self):
        """Update statistics display."""
        self.total_activations_label.setText(f"🎯 Total Activations: {self.total_activations}")
        self.successful_transcriptions_label.setText(f"✅ Successful Transcriptions: {self.successful_transcriptions}")
        self.successful_parses_label.setText(f"🤖 Successful Parses: {self.successful_parses}")
        
        # Calculate rates
        if self.total_activations > 0:
            transcription_rate = (self.successful_transcriptions / self.total_activations) * 100
            self.transcription_rate_label.setText(f"📊 Transcription Rate: {transcription_rate:.1f}%")
        else:
            self.transcription_rate_label.setText("📊 Transcription Rate: 0%")
        
        if self.successful_transcriptions > 0:
            parsing_rate = (self.successful_parses / self.successful_transcriptions) * 100
            self.parsing_rate_label.setText(f"🎯 Parsing Rate: {parsing_rate:.1f}%")
        else:
            self.parsing_rate_label.setText("🎯 Parsing Rate: 0%")
    
    def test_command_parsing(self):
        """Test command parsing with sample commands."""
        test_commands = [
            "describe data",
            "filter age greater than 25",
            "sort by name ascending",
            "visualize column sales",
            "sum of revenue"
        ]
        
        self.add_debug_log("🧪 Testing command parsing...")
        
        for cmd in test_commands:
            if self.command_parser:
                result = self.command_parser.parse(cmd)
                self.add_debug_log(f"📝 '{cmd}' → {result.intent.value} (confidence: {result.confidence:.2f})")
                if result.intent != CommandIntent.UNKNOWN:
                    details = []
                    if result.column:
                        details.append(f"column={result.column}")
                    if result.operation:
                        details.append(f"op={result.operation.value}")
                    if result.value:
                        details.append(f"value={result.value}")
                    if details:
                        self.add_debug_log(f"   └─ {', '.join(details)}")
    
    # Copy all the other methods from the original VoicePanel but with debug logging
    def update_gesture_status(self, gesture: str, confidence: float = 0.0):
        """Update current gesture status from camera panel."""
        self.current_gesture = gesture if confidence > 0.7 else None
        
        if self.current_gesture:
            self.gesture_status_label.setText(f"✋ Gesture: {gesture.replace('_', ' ').title()} ({confidence:.2f})")
            self.gesture_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
            if gesture == self.trigger_gesture:
                self.add_debug_log(f"✌️ Trigger gesture detected: {gesture} (confidence: {confidence:.2f})")
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
            if self.gesture_start_time is None:
                self.gesture_start_time = current_time
                self.activation_progress.setVisible(True)
                self.activation_progress.setValue(0)
                self.add_debug_log("⏱️ Gesture activation started - hold for 1 second")
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
                self.add_debug_log("❌ Gesture lost - activation cancelled")
    
    def activate_voice_recognition(self):
        """Activate voice recognition through gesture trigger."""
        if self.is_listening:
            return
        
        self.total_activations += 1
        self.update_statistics_display()
        
        self.gesture_trigger_detected.emit(self.trigger_gesture)
        self.add_debug_log("🎯 Gesture activation triggered!")
        
        if self.voice_handler and self.voice_handler.start_listening():
            self.is_listening = True
            self.recording_start_time = time.time()
            self.voice_status_label.setText("🎤 Listening...")
            self.voice_status_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
            self.add_debug_log("🎤 Voice recognition started - speak now!")
            logger.info("Voice recognition activated by gesture")
        else:
            self.add_debug_log("❌ Failed to start voice recognition")
    
    def toggle_manual_listening(self):
        """Toggle manual voice recognition."""
        if not VOICE_AVAILABLE or not self.voice_handler:
            return
        
        if self.is_listening:
            self.voice_handler.stop_listening()
            self.is_listening = False
            self.manual_button.setText("🎤 Start Voice Recognition")
            self.voice_status_label.setText("✅ Voice Recognition: Ready")
            self.voice_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.add_debug_log("🛑 Manual voice recognition stopped")
        else:
            self.total_activations += 1
            self.update_statistics_display()
            
            if self.voice_handler.start_listening():
                self.is_listening = True
                self.recording_start_time = time.time()
                self.manual_button.setText("🛑 Stop Listening")
                self.voice_status_label.setText("🎤 Listening...")
                self.voice_status_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
                self.add_debug_log("🎤 Manual voice recognition started - speak now!")
    
    def toggle_voice_recognition(self, state):
        """Enable/disable voice recognition."""
        self.is_voice_active = state == Qt.Checked
        if not self.is_voice_active and self.is_listening:
            self.voice_handler.stop_listening()
            self.is_listening = False
            self.manual_button.setText("🎤 Start Voice Recognition")
        
        status = "enabled" if self.is_voice_active else "disabled"
        self.add_debug_log(f"🔧 Voice recognition {status}")
    
    def test_microphone(self):
        """Test microphone functionality."""
        if not VOICE_AVAILABLE:
            self.add_debug_log("❌ Voice components not available")
            return
        
        self.add_debug_log("🔊 Testing microphone...")
        
        try:
            if test_microphone():
                self.add_debug_log("✅ Microphone test successful - audio detected")
            else:
                self.add_debug_log("❌ Microphone test failed - no audio detected")
        except Exception as e:
            self.add_debug_log(f"❌ Microphone test error: {str(e)}")
    
    @pyqtSlot(str, float)
    def on_transcription_received(self, text: str, confidence: float):
        """Handle transcription from voice recognition."""
        self.successful_transcriptions += 1
        self.update_statistics_display()
        
        self.add_debug_log(f"🎤 TRANSCRIBED: '{text}' (confidence: {confidence:.2f})")
        self.transcription_status_label.setText(f"💬 Last Transcription: {text}")
        self.transcription_status_label.setStyleSheet("color: #00b894; font-weight: bold;")
        
        # Parse the command
        if self.command_parser:
            parsed_command = self.command_parser.parse(text)
            self.add_debug_log(f"🤖 PARSED: Intent={parsed_command.intent.value}, Confidence={parsed_command.confidence:.2f}")
            
            if parsed_command.intent != CommandIntent.UNKNOWN:
                self.successful_parses += 1
                self.update_statistics_display()
                
                # Add command details
                details = []
                if parsed_command.column:
                    details.append(f"column={parsed_command.column}")
                if parsed_command.operation:
                    details.append(f"operation={parsed_command.operation.value}")
                if parsed_command.value:
                    details.append(f"value={parsed_command.value}")
                if parsed_command.chart_type:
                    details.append(f"chart={parsed_command.chart_type.value}")
                if parsed_command.sort_direction:
                    details.append(f"direction={parsed_command.sort_direction.value}")
                
                if details:
                    self.add_debug_log(f"   └─ {', '.join(details)}")
                
                self.parsing_status_label.setText(f"🤖 Last Parse: {parsed_command.intent.value}")
                self.parsing_status_label.setStyleSheet("color: #00b894; font-weight: bold;")
                
                # Emit parsed command
                self.command_parsed.emit(parsed_command.to_dict())
                self.add_debug_log("✅ Command successfully parsed and emitted!")
            else:
                self.add_debug_log("⚠️ Command not recognized - check command patterns")
                self.parsing_status_label.setText("🤖 Last Parse: UNKNOWN")
                self.parsing_status_label.setStyleSheet("color: #e17055; font-weight: bold;")
        
        # Reset listening state
        self.is_listening = False
        self.manual_button.setText("🎤 Start Voice Recognition")
        self.voice_status_label.setText("✅ Voice Recognition: Ready")
        self.voice_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.last_transcription_time = time.time()
    
    @pyqtSlot(str)
    def on_voice_error(self, error_message: str):
        """Handle voice recognition errors."""
        self.add_debug_log(f"❌ VOICE ERROR: {error_message}")
        
        # Reset listening state
        self.is_listening = False
        self.manual_button.setText("🎤 Start Voice Recognition")
        self.voice_status_label.setText("✅ Voice Recognition: Ready")
        self.voice_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
    
    @pyqtSlot(bool)
    def on_recording_status_changed(self, is_recording: bool):
        """Handle recording status changes."""
        if is_recording:
            self.voice_status_label.setText("🔴 Recording...")
            self.voice_status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            self.add_debug_log("🔴 Audio recording started")
        else:
            self.voice_status_label.setText("⏳ Processing...")
            self.voice_status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
            self.add_debug_log("⏳ Audio recording stopped - processing with Whisper...")
    
    def add_debug_log(self, message: str):
        """Add message to debug logs."""
        timestamp = time.strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        self.debug_logs.append(f"[{timestamp}] {message}")
        
        # Auto-scroll to bottom if enabled
        if self.auto_scroll_checkbox.isChecked():
            cursor = self.debug_logs.textCursor()
            cursor.movePosition(cursor.End)
            self.debug_logs.setTextCursor(cursor)
    
    def cleanup(self):
        """Clean up voice components."""
        if self.voice_handler:
            self.voice_handler.cleanup()
        
        if hasattr(self, 'gesture_timer'):
            self.gesture_timer.stop()
        
        if hasattr(self, 'audio_timer'):
            self.audio_timer.stop()

# For testing the debug voice panel standalone
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
    
    debug_panel = VoiceDebugPanel()
    debug_panel.show()
    
    sys.exit(app.exec_()) 