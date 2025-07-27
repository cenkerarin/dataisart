"""
Example AI Assistant Application
===============================

Example application demonstrating the complete AI assistant system
with voice recognition and data analysis capabilities.
"""

import sys
import os
from pathlib import Path
import logging

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "config"))

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QPushButton, QLabel, QFileDialog, QTextEdit, QSplitter
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Import our components
from ai_assistant import AIAssistant
from ui_integration import AIResultDisplayWidget
from voice_ai_bridge import VoiceAIIntegration

# Import configuration
try:
    from app.settings import LLM_CONFIG, VOICE_CONFIG, DATA_CONFIG
except ImportError:
    # Fallback configuration
    LLM_CONFIG = {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "api_key": "",
        "temperature": 0.1,
        "max_tokens": 1000,
        "fallback_to_patterns": True,
    }
    VOICE_CONFIG = {
        "recognition_timeout": 5,
        "phrase_timeout": 1,
    }
    DATA_CONFIG = {
        "supported_formats": [".csv", ".xlsx", ".json"],
        "max_file_size_mb": 100,
    }

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIAssistantDemo(QMainWindow):
    """Main demo application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤖 AI Assistant Demo - Voice-Controlled Data Analysis")
        self.setGeometry(100, 100, 1400, 900)
        
        # Configuration
        self.config = {
            "llm": LLM_CONFIG,
            "voice": VOICE_CONFIG,
            "data": DATA_CONFIG
        }
        
        # Initialize AI integration
        self.ai_integration = VoiceAIIntegration(self.config)
        
        # Setup UI
        self.setup_ui()
        
        # Initialize components
        self.initialize_components()
        
    def setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Content area
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Controls
        controls_panel = self.create_controls_panel()
        content_splitter.addWidget(controls_panel)
        
        # Right panel - AI Results
        self.results_display = AIResultDisplayWidget()
        content_splitter.addWidget(self.results_display)
        
        # Set splitter proportions
        content_splitter.setSizes([400, 1000])
        
        main_layout.addWidget(content_splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready - Load a dataset and start voice commands")
        
        # Apply dark theme
        self.apply_dark_theme()
    
    def create_header(self) -> QWidget:
        """Create the header section."""
        header = QWidget()
        layout = QHBoxLayout(header)
        
        title = QLabel("🤖 AI Assistant Demo")
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2196F3;
                padding: 15px;
            }
        """)
        
        subtitle = QLabel("Voice-controlled data analysis and visualization")
        subtitle.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #888;
                padding: 5px 15px;
            }
        """)
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch()
        
        return header
    
    def create_controls_panel(self) -> QWidget:
        """Create the controls panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Dataset section
        dataset_section = QLabel("📁 Dataset")
        dataset_section.setStyleSheet("font-weight: bold; font-size: 16px; padding: 10px;")
        layout.addWidget(dataset_section)
        
        self.load_button = QPushButton("Load Dataset")
        self.load_button.clicked.connect(self.load_dataset)
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        layout.addWidget(self.load_button)
        
        self.dataset_info = QLabel("No dataset loaded")
        self.dataset_info.setWordWrap(True)
        self.dataset_info.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 5px;
                margin: 5px 0;
            }
        """)
        layout.addWidget(self.dataset_info)
        
        # Voice section
        voice_section = QLabel("🎤 Voice Commands")
        voice_section.setStyleSheet("font-weight: bold; font-size: 16px; padding: 10px;")
        layout.addWidget(voice_section)
        
        self.voice_button = QPushButton("Start Voice Recognition")
        self.voice_button.clicked.connect(self.toggle_voice_recognition)
        self.voice_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        layout.addWidget(self.voice_button)
        
        # Text input for testing
        text_section = QLabel("⌨️ Text Commands (for testing)")
        text_section.setStyleSheet("font-weight: bold; font-size: 16px; padding: 10px;")
        layout.addWidget(text_section)
        
        self.text_input = QTextEdit()
        self.text_input.setMaximumHeight(100)
        self.text_input.setPlaceholderText("Type a command here (e.g., 'describe data')")
        layout.addWidget(self.text_input)
        
        self.process_button = QPushButton("Process Command")
        self.process_button.clicked.connect(self.process_text_command)
        self.process_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        layout.addWidget(self.process_button)
        
        # Examples section
        examples_section = QLabel("💡 Example Commands")
        examples_section.setStyleSheet("font-weight: bold; font-size: 16px; padding: 10px;")
        layout.addWidget(examples_section)
        
        self.examples_list = QLabel()
        self.examples_list.setWordWrap(True)
        self.examples_list.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: rgba(255, 193, 7, 0.1);
                border: 1px solid #FFC107;
                border-radius: 5px;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.examples_list)
        
        layout.addStretch()
        
        return panel
    
    def apply_dark_theme(self):
        """Apply dark theme to the application."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QSplitter::handle {
                background-color: #555555;
                width: 3px;
            }
        """)
    
    def initialize_components(self):
        """Initialize AI components."""
        try:
            # Initialize AI integration
            success = self.ai_integration.initialize()
            
            if success:
                self.statusBar().showMessage("✅ AI Assistant initialized successfully")
                
                # Set up UI integration
                self.ai_integration.set_ui_display(self.results_display)
                
                # Update examples
                self.update_examples()
                
            else:
                self.statusBar().showMessage("⚠️ AI Assistant initialized with limited functionality")
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            self.statusBar().showMessage(f"❌ Initialization error: {str(e)}")
    
    def load_dataset(self):
        """Load a dataset file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Dataset",
            "",
            "Data Files (*.csv *.xlsx *.json);;CSV Files (*.csv);;Excel Files (*.xlsx);;JSON Files (*.json)"
        )
        
        if file_path:
            try:
                result = self.ai_integration.load_dataset(file_path)
                
                if result.success:
                    self.dataset_info.setText(result.message)
                    self.statusBar().showMessage(f"✅ Dataset loaded: {Path(file_path).name}")
                    self.update_examples()
                else:
                    self.dataset_info.setText(f"Error: {result.error}")
                    self.statusBar().showMessage(f"❌ Failed to load dataset: {result.error}")
                    
            except Exception as e:
                logger.error(f"Error loading dataset: {e}")
                self.statusBar().showMessage(f"❌ Error loading dataset: {str(e)}")
    
    def toggle_voice_recognition(self):
        """Toggle voice recognition on/off."""
        try:
            status = self.ai_integration.bridge.get_status()
            
            if status["voice_active"]:
                self.ai_integration.stop_voice_recognition()
                self.voice_button.setText("Start Voice Recognition")
                self.voice_button.setStyleSheet(self.voice_button.styleSheet().replace("#1976D2", "#2196F3"))
                self.statusBar().showMessage("🔇 Voice recognition stopped")
            else:
                if not status["voice_available"]:
                    self.statusBar().showMessage("❌ Voice recognition not available")
                    return
                
                self.ai_integration.start_voice_recognition()
                self.voice_button.setText("Stop Voice Recognition")
                self.voice_button.setStyleSheet(self.voice_button.styleSheet().replace("#2196F3", "#1976D2"))
                self.statusBar().showMessage("🎤 Voice recognition active - Speak your commands!")
                
        except Exception as e:
            logger.error(f"Error toggling voice recognition: {e}")
            self.statusBar().showMessage(f"❌ Voice error: {str(e)}")
    
    def process_text_command(self):
        """Process text command from input field."""
        command = self.text_input.toPlainText().strip()
        if not command:
            return
        
        try:
            self.statusBar().showMessage(f"🤖 Processing: {command}")
            result = self.ai_integration.process_text_command(command)
            
            if result.success:
                self.statusBar().showMessage(f"✅ Command completed: {result.action_type.value}")
            else:
                self.statusBar().showMessage(f"❌ Command failed: {result.error}")
            
            # Clear input
            self.text_input.clear()
            
        except Exception as e:
            logger.error(f"Error processing text command: {e}")
            self.statusBar().showMessage(f"❌ Processing error: {str(e)}")
    
    def update_examples(self):
        """Update example commands based on current state."""
        try:
            examples = self.ai_integration.get_example_commands()
            examples_text = "\n".join([f"• {cmd}" for cmd in examples])
            self.examples_list.setText(examples_text)
        except Exception as e:
            logger.error(f"Error updating examples: {e}")

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("AI Assistant Demo")
    app.setApplicationVersion("1.0.0")
    
    # Create and show main window
    window = AIAssistantDemo()
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 