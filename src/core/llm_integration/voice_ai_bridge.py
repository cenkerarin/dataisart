"""
Voice-AI Bridge
===============

Bridge module that connects voice recognition with AI assistant.
Handles the flow from voice command → parsing → AI processing → UI display.
"""

import logging
import threading
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import sys

# Add voice recognition path for imports
voice_path = Path(__file__).parent.parent / "voice_recognition"
sys.path.insert(0, str(voice_path))

try:
    from .ai_assistant import AIAssistant, ActionResult, ActionType
    from .ui_integration import AICommandProcessor
except ImportError:
    from ai_assistant import AIAssistant, ActionResult, ActionType
    from ui_integration import AICommandProcessor

logger = logging.getLogger(__name__)

class VoiceAIBridge:
    """Bridge between voice recognition and AI assistant."""
    
    def __init__(self, ai_assistant: AIAssistant, config: Dict[str, Any]):
        """
        Initialize the voice-AI bridge.
        
        Args:
            ai_assistant: The AI assistant instance
            config: Configuration dictionary
        """
        self.ai_assistant = ai_assistant
        self.config = config
        self.is_active = False
        
        # Voice components
        self.voice_handler = None
        self.command_parser = None
        
        # Callbacks
        self.result_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None
        
        # State
        self.last_command = ""
        self.processing = False
        
    def initialize_voice_components(self) -> bool:
        """Initialize voice recognition components."""
        try:
            # Import voice components
            from voice_handler import VoiceHandler
            from command_parser import create_parser
            
            # Initialize voice handler
            self.voice_handler = VoiceHandler(self.config.get("voice", {}))
            if not self.voice_handler.initialize():
                logger.error("Failed to initialize voice handler")
                return False
            
            # Initialize command parser
            self.command_parser = create_parser()
            
            # Set up voice handler callback
            self.voice_handler.set_transcription_callback(self._handle_transcription)
            
            logger.info("Voice components initialized successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Voice recognition not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Error initializing voice components: {e}")
            return False
    
    def set_result_callback(self, callback: Callable[[ActionResult], None]):
        """Set callback for when AI processing is complete."""
        self.result_callback = callback
    
    def set_status_callback(self, callback: Callable[[str], None]):
        """Set callback for status updates."""
        self.status_callback = callback
    
    def start_listening(self):
        """Start voice recognition."""
        if self.voice_handler and not self.is_active:
            self.is_active = True
            self.voice_handler.start_listening()
            self._update_status("🎤 Listening for commands...")
            logger.info("Started voice listening")
    
    def stop_listening(self):
        """Stop voice recognition."""
        if self.voice_handler and self.is_active:
            self.is_active = False
            self.voice_handler.stop_listening()
            self._update_status("⏹️ Voice recognition stopped")
            logger.info("Stopped voice listening")
    
    def process_text_command(self, command: str) -> ActionResult:
        """Process a text command directly (for testing)."""
        if self.processing:
            return ActionResult(
                success=False,
                action_type=ActionType.UNKNOWN,
                error="Already processing a command"
            )
        
        self.processing = True
        self._update_status(f"🤖 Processing: {command}")
        
        try:
            # Parse command if parser is available
            parsed_command = None
            if self.command_parser:
                parsed_command = self.command_parser.parse(command)
                if parsed_command:
                    parsed_command = parsed_command.to_dict()
            
            # Process with AI assistant
            result = self.ai_assistant.process_voice_command(command, parsed_command)
            
            # Update status based on result
            if result.success:
                self._update_status(f"✅ Completed: {result.action_type.value}")
            else:
                self._update_status(f"❌ Error: {result.error}")
            
            # Call result callback
            if self.result_callback:
                self.result_callback(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing text command: {e}")
            error_result = ActionResult(
                success=False,
                action_type=ActionType.UNKNOWN,
                error=str(e)
            )
            
            if self.result_callback:
                self.result_callback(error_result)
                
            return error_result
            
        finally:
            self.processing = False
    
    def _handle_transcription(self, transcription: str, confidence: float = 0.0):
        """Handle voice transcription from voice handler."""
        if not transcription.strip():
            return
        
        logger.info(f"Voice transcription: {transcription} (confidence: {confidence})")
        
        # Update last command
        self.last_command = transcription
        
        # Process in separate thread to avoid blocking
        threading.Thread(
            target=self._process_voice_command_async,
            args=(transcription,),
            daemon=True
        ).start()
    
    def _process_voice_command_async(self, command: str):
        """Process voice command asynchronously."""
        try:
            result = self.process_text_command(command)
            logger.info(f"Voice command processed: {result.success}")
        except Exception as e:
            logger.error(f"Error in async voice processing: {e}")
    
    def _update_status(self, status: str):
        """Update status via callback."""
        if self.status_callback:
            self.status_callback(status)
        logger.debug(f"Status: {status}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bridge status."""
        return {
            "voice_active": self.is_active,
            "processing": self.processing,
            "last_command": self.last_command,
            "voice_available": self.voice_handler is not None,
            "ai_initialized": self.ai_assistant.is_initialized,
            "dataset_loaded": self.ai_assistant.dataset_context.dataframe is not None
        }

class VoiceAIIntegration:
    """Complete integration of voice recognition with AI assistant."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the complete voice-AI integration."""
        self.config = config
        
        # Initialize AI assistant
        self.ai_assistant = AIAssistant(config)
        
        # Initialize bridge
        self.bridge = VoiceAIBridge(self.ai_assistant, config)
        
        # Initialize command processor
        self.command_processor = AICommandProcessor(self.ai_assistant)
        
        # UI components (set later)
        self.ui_display = None
        
    def initialize(self) -> bool:
        """Initialize all components."""
        try:
            # Initialize AI assistant
            self.ai_assistant.initialize()
            
            # Initialize voice components
            self.bridge.initialize_voice_components()
            
            # Set up callbacks
            self.bridge.set_result_callback(self._handle_ai_result)
            self.bridge.set_status_callback(self._handle_status_update)
            
            logger.info("Voice-AI integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing voice-AI integration: {e}")
            return False
    
    def set_ui_display(self, display_widget):
        """Set the UI display widget for results."""
        self.ui_display = display_widget
        
        # Connect UI signals
        if hasattr(display_widget, 'action_requested'):
            display_widget.action_requested.connect(self.process_text_command)
    
    def load_dataset(self, file_path: str, name: str = "") -> ActionResult:
        """Load a dataset for analysis."""
        return self.ai_assistant.load_dataset(file_path, name)
    
    def start_voice_recognition(self):
        """Start voice recognition."""
        self.bridge.start_listening()
    
    def stop_voice_recognition(self):
        """Stop voice recognition."""
        self.bridge.stop_listening()
    
    def process_text_command(self, command: str) -> ActionResult:
        """Process a text command."""
        return self.bridge.process_text_command(command)
    
    def _handle_ai_result(self, result: ActionResult):
        """Handle AI result and display in UI."""
        if self.ui_display:
            self.ui_display.display_result(result)
        
        logger.info(f"AI result: {result.action_type.value} - Success: {result.success}")
    
    def _handle_status_update(self, status: str):
        """Handle status updates."""
        logger.info(f"Status update: {status}")
        
        # You can emit signals here for UI status updates
        # if hasattr(self, 'status_updated'):
        #     self.status_updated.emit(status)
    
 