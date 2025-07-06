"""
Voice Handler
============

Handles voice recognition and speech-to-text conversion.
"""

import speech_recognition as sr
import whisper
import logging
from typing import Optional, Dict, Any, Callable
import threading
import queue

logger = logging.getLogger(__name__)

class VoiceHandler:
    """Handles voice recognition and command processing."""
    
    def __init__(self):
        """Initialize the voice handler."""
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.whisper_model = None
        self.is_listening = False
        self.command_queue = queue.Queue()
        self.callback_function: Optional[Callable] = None
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize voice recognition components.
        
        Args:
            config (Dict[str, Any]): Voice recognition configuration
            
        Returns:
            bool: True if successful
        """
        try:
            # Initialize microphone
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
            
            # Set recognition parameters
            self.recognizer.energy_threshold = config.get("energy_threshold", 4000)
            self.recognizer.dynamic_energy_threshold = config.get("dynamic_energy_threshold", True)
            self.recognizer.pause_threshold = config.get("pause_threshold", 0.8)
            
            # Initialize Whisper model (optional)
            try:
                self.whisper_model = whisper.load_model("base")
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load Whisper model: {str(e)}")
            
            logger.info("Voice handler initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing voice handler: {str(e)}")
            return False
    
    def start_listening(self, callback_function: Callable[[str], None]) -> bool:
        """
        Start continuous voice recognition.
        
        Args:
            callback_function (Callable): Function to call with recognized text
            
        Returns:
            bool: True if started successfully
        """
        if self.is_listening:
            logger.warning("Already listening")
            return False
        
        self.callback_function = callback_function
        self.is_listening = True
        
        # Start listening in a separate thread
        listening_thread = threading.Thread(target=self._listen_continuously)
        listening_thread.daemon = True
        listening_thread.start()
        
        logger.info("Started continuous voice recognition")
        return True
    
    def stop_listening(self):
        """Stop continuous voice recognition."""
        self.is_listening = False
        logger.info("Stopped continuous voice recognition")
    
    def _listen_continuously(self):
        """Continuously listen for voice commands."""
        while self.is_listening:
            try:
                # Listen for audio
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
                # Process audio
                text = self._process_audio(audio)
                
                if text and self.callback_function:
                    self.callback_function(text)
                    
            except sr.WaitTimeoutError:
                # Continue listening
                pass
            except Exception as e:
                logger.error(f"Error during continuous listening: {str(e)}")
    
    def _process_audio(self, audio) -> Optional[str]:
        """
        Process audio and convert to text.
        
        Args:
            audio: Audio data from microphone
            
        Returns:
            Optional[str]: Recognized text or None
        """
        try:
            # Try Google Speech Recognition first
            text = self.recognizer.recognize_google(audio)
            logger.info(f"Google Speech Recognition: {text}")
            return text
            
        except sr.UnknownValueError:
            logger.debug("Google Speech Recognition could not understand audio")
            
        except sr.RequestError as e:
            logger.error(f"Google Speech Recognition error: {str(e)}")
        
        # Fallback to Whisper if available
        if self.whisper_model:
            try:
                # Convert audio to format Whisper can process
                audio_data = sr.AudioData(audio.get_raw_data(), audio.sample_rate, audio.sample_width)
                result = self.whisper_model.transcribe(audio_data)
                text = result["text"].strip()
                logger.info(f"Whisper transcription: {text}")
                return text
                
            except Exception as e:
                logger.error(f"Whisper transcription error: {str(e)}")
        
        return None
    
    def recognize_once(self, timeout: int = 5) -> Optional[str]:
        """
        Perform single voice recognition.
        
        Args:
            timeout (int): Maximum time to wait for speech
            
        Returns:
            Optional[str]: Recognized text or None
        """
        try:
            with self.microphone as source:
                logger.info("Listening for voice command...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
            
            return self._process_audio(audio)
            
        except sr.WaitTimeoutError:
            logger.info("No speech detected within timeout")
            return None
        except Exception as e:
            logger.error(f"Error during single recognition: {str(e)}")
            return None
    
    def process_command(self, text: str) -> Dict[str, Any]:
        """
        Process voice command and extract intent.
        
        Args:
            text (str): Recognized text
            
        Returns:
            Dict[str, Any]: Command information
        """
        text = text.lower().strip()
        
        # Basic command parsing (placeholder)
        command_info = {
            "text": text,
            "intent": "unknown",
            "entities": [],
            "confidence": 0.0
        }
        
        # Simple intent detection (placeholder)
        if any(word in text for word in ["visualize", "plot", "chart", "graph"]):
            command_info["intent"] = "visualize"
        elif any(word in text for word in ["analyze", "analysis", "correlation"]):
            command_info["intent"] = "analyze"
        elif any(word in text for word in ["select", "choose", "pick"]):
            command_info["intent"] = "select"
        elif any(word in text for word in ["load", "open", "import"]):
            command_info["intent"] = "load_data"
        
        return command_info
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_listening()
        logger.info("Voice handler cleaned up") 