"""
Voice Handler
============

Advanced voice recognition handler that records audio segments, processes them with Whisper,
and provides non-blocking operation for PyQt applications.
"""

import os
import sys
import time
import threading
import queue
import logging
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from pathlib import Path
import tempfile

# Audio processing
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal

# Whisper for transcription
import whisper

# PyQt for threading and signals
try:
    from PyQt5.QtCore import QThread, pyqtSignal, QObject, QTimer
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    # Fallback to standard threading
    import threading

logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Audio recording configuration."""
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = 'float32'  # Use float32 for proper normalization
    chunk_duration: float = 0.1  # seconds
    silence_threshold: float = 0.02  # Adjusted for normalized audio
    silence_duration: float = 1.0  # seconds of silence before stopping
    max_recording_duration: float = 30.0  # maximum recording length
    min_recording_duration: float = 0.5   # minimum recording length

@dataclass
class WhisperConfig:
    """Whisper model configuration."""
    model_name: str = "base"
    device: str = "cpu"  # or "cuda" if available
    language: Optional[str] = None  # auto-detect if None
    temperature: float = 0.0

class AudioSegment:
    """Represents a recorded audio segment."""
    
    def __init__(self, audio_data: np.ndarray, sample_rate: int, timestamp: float):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.timestamp = timestamp
        self.duration = len(audio_data) / sample_rate
        self.temp_file: Optional[str] = None
    
    def save_to_file(self, filepath: Optional[str] = None) -> str:
        """Save audio segment to WAV file."""
        if filepath is None:
            # Create temporary file
            temp_fd, filepath = tempfile.mkstemp(suffix='.wav', prefix='audio_segment_')
            os.close(temp_fd)
            self.temp_file = filepath
        
        # Ensure audio data is properly normalized
        if self.audio_data.dtype == np.float32:
            # Audio is already normalized to [-1, 1], convert to int16
            audio_normalized = np.clip(self.audio_data * 32767, -32768, 32767).astype(np.int16)
        elif self.audio_data.dtype == np.int16:
            # Audio is already int16
            audio_normalized = self.audio_data
        else:
            # Assume it needs normalization
            max_val = np.max(np.abs(self.audio_data))
            if max_val > 1.0:
                normalized = self.audio_data / max_val
            else:
                normalized = self.audio_data
            audio_normalized = np.clip(normalized * 32767, -32768, 32767).astype(np.int16)
        
        wavfile.write(filepath, self.sample_rate, audio_normalized)
        return filepath
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_file and os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
                self.temp_file = None
            except OSError as e:
                logger.warning(f"Could not remove temporary file {self.temp_file}: {e}")

class VoiceRecorder:
    """Handles audio recording with silence detection."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.is_recording = False
        self.audio_buffer = []
        self.silence_counter = 0
        self.recording_start_time = 0
        
    def start_recording(self) -> bool:
        """Start recording audio."""
        try:
            # Check if sounddevice can access microphone
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            
            if default_input is None:
                logger.error("No default input device found")
                return False
            
            logger.info(f"Using input device: {devices[default_input]['name']}")
            
            self.is_recording = True
            self.audio_buffer = []
            self.silence_counter = 0
            self.recording_start_time = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return False
    
    def stop_recording(self) -> Optional[AudioSegment]:
        """Stop recording and return audio segment."""
        if not self.is_recording:
            return None
        
        self.is_recording = False
        
        if not self.audio_buffer:
            return None
        
        # Combine audio chunks
        audio_data = np.concatenate(self.audio_buffer, axis=0)
        
        # Check minimum duration
        duration = len(audio_data) / self.config.sample_rate
        if duration < self.config.min_recording_duration:
            logger.debug(f"Recording too short: {duration:.2f}s")
            return None
        
        return AudioSegment(audio_data, self.config.sample_rate, self.recording_start_time)
    
    def process_audio_chunk(self, chunk: np.ndarray) -> bool:
        """
        Process audio chunk and detect silence.
        
        Returns:
            bool: True if should continue recording, False if should stop
        """
        if not self.is_recording:
            return False
        
        self.audio_buffer.append(chunk.copy())
        
        # Calculate RMS for silence detection
        rms = np.sqrt(np.mean(chunk**2))
        
        if rms < self.config.silence_threshold:
            self.silence_counter += 1
        else:
            self.silence_counter = 0
        
        # Check stopping conditions
        current_duration = time.time() - self.recording_start_time
        silence_duration = self.silence_counter * self.config.chunk_duration
        
        # Debug logging (every second or so)
        chunk_count = len(self.audio_buffer)
        if chunk_count % int(1.0 / self.config.chunk_duration) == 0:  # Every ~1 second
            logger.debug(f"Recording: {current_duration:.1f}s, RMS: {rms:.4f}, Silence: {silence_duration:.1f}s")
        
        # Stop if max duration reached or silence detected
        if current_duration > self.config.max_recording_duration:
            logger.info(f"Stopping recording: max duration reached ({current_duration:.1f}s)")
            return False
        elif silence_duration > self.config.silence_duration:
            logger.info(f"Stopping recording: silence detected ({silence_duration:.1f}s)")
            return False
        
        return True

if PYQT_AVAILABLE:
    class VoiceHandlerThread(QThread):
        """PyQt thread for voice processing."""
        
        transcription_ready = pyqtSignal(str, float)  # text, confidence
        error_occurred = pyqtSignal(str)
        recording_started = pyqtSignal()
        recording_stopped = pyqtSignal()
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self.audio_config = AudioConfig()
            self.whisper_config = WhisperConfig()
            self.recorder = VoiceRecorder(self.audio_config)
            self.whisper_model = None
            self.is_active = False
            self.retry_count = 0
            self.max_retries = 3
            
        def initialize_whisper(self) -> bool:
            """Initialize Whisper model."""
            try:
                logger.info(f"Loading Whisper model: {self.whisper_config.model_name}")
                self.whisper_model = whisper.load_model(
                    self.whisper_config.model_name,
                    device=self.whisper_config.device
                )
                logger.info("Whisper model loaded successfully")
                return True
            except Exception as e:
                error_msg = f"Failed to load Whisper model: {e}"
                logger.error(error_msg)
                self.error_occurred.emit(error_msg)
                return False
        
        def run(self):
            """Main thread execution."""
            if not self.initialize_whisper():
                return
            
            self.is_active = True
            
            while self.is_active:
                try:
                    self._record_and_transcribe()
                except Exception as e:
                    logger.error(f"Error in voice processing: {e}")
                    self.error_occurred.emit(f"Voice processing error: {e}")
                    
                    # Retry mechanism
                    self.retry_count += 1
                    if self.retry_count >= self.max_retries:
                        logger.error("Max retries reached, stopping voice handler")
                        break
                    
                    # Wait before retrying
                    time.sleep(1.0)
        
        def _record_and_transcribe(self):
            """Record audio and transcribe."""
            # Start recording
            if not self.recorder.start_recording():
                self.error_occurred.emit("Failed to start recording")
                return
            
            self.recording_started.emit()
            logger.info("Started recording...")
            
            try:
                # Reset callback counter for debugging
                self._callback_count = 0
                
                # Record audio stream
                with sd.InputStream(
                    samplerate=self.audio_config.sample_rate,
                    channels=self.audio_config.channels,
                    dtype=self.audio_config.dtype,
                    blocksize=int(self.audio_config.sample_rate * self.audio_config.chunk_duration),
                    callback=self._audio_callback
                ):
                    # Wait for recording to complete
                    start_time = time.time()
                    while self.recorder.is_recording and self.is_active:
                        time.sleep(0.1)
                        # Debug: Show recording progress
                        elapsed = time.time() - start_time
                        if elapsed > 0 and int(elapsed) % 5 == 0 and elapsed - int(elapsed) < 0.1:
                            logger.info(f"Recording... {elapsed:.1f}s elapsed")
                
            except Exception as e:
                logger.error(f"Recording error: {e}")
                self.error_occurred.emit(f"Recording error: {e}")
                return
            
            # Stop recording and get segment
            segment = self.recorder.stop_recording()
            self.recording_stopped.emit()
            
            if segment is None:
                logger.warning("No audio segment to process - recording may have been too short or silent")
                return
            
            logger.info(f"Audio segment captured: {segment.duration:.2f}s duration")
            
            # Transcribe audio
            self._transcribe_segment(segment)
        
        def _audio_callback(self, indata, frames, time, status):
            """Audio stream callback."""
            if status:
                logger.warning(f"Audio callback status: {status}")
            
            # Convert to correct format
            audio_chunk = indata[:, 0] if indata.ndim > 1 else indata
            
            # Debug: Log audio levels periodically
            if hasattr(self, '_callback_count'):
                self._callback_count += 1
            else:
                self._callback_count = 1
            
            if self._callback_count % 50 == 0:  # Every ~0.5 seconds at 0.1s chunks
                rms = np.sqrt(np.mean(audio_chunk**2))
                logger.debug(f"Audio level: {rms:.4f}")
            
            # Process chunk and check if should continue
            if not self.recorder.process_audio_chunk(audio_chunk):
                # Signal to stop recording
                logger.debug("Stopping recording due to silence or max duration")
                pass
        
        def _transcribe_segment(self, segment: AudioSegment):
            """Transcribe audio segment using Whisper."""
            try:
                # Save segment to temporary file
                temp_file = segment.save_to_file()
                
                logger.info(f"Transcribing audio segment ({segment.duration:.2f}s)")
                
                # Transcribe with Whisper
                result = self.whisper_model.transcribe(
                    temp_file,
                    language=self.whisper_config.language,
                    temperature=self.whisper_config.temperature
                )
                
                text = result["text"].strip()
                
                # Calculate confidence (Whisper doesn't provide direct confidence)
                # Use length and presence of text as proxy
                confidence = min(1.0, len(text) / 100.0) if text else 0.0
                
                if text:
                    logger.info(f"Transcription: {text}")
                    self.transcription_ready.emit(text, confidence)
                    self.retry_count = 0  # Reset retry counter on success
                
                # Cleanup
                segment.cleanup()
                
            except Exception as e:
                error_msg = f"Transcription error: {e}"
                logger.error(error_msg)
                self.error_occurred.emit(error_msg)
                segment.cleanup()
        
        def stop(self):
            """Stop the voice handler."""
            self.is_active = False
            self.recorder.is_recording = False

class VoiceHandler(QObject if PYQT_AVAILABLE else object):
    """Main voice handler class with PyQt integration."""
    
    if PYQT_AVAILABLE:
        transcription_received = pyqtSignal(str, float)
        error_occurred = pyqtSignal(str)
        recording_status_changed = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        if PYQT_AVAILABLE:
            super().__init__(parent)
        
        self.thread = None
        self.callback_function: Optional[Callable[[str, float], None]] = None
        self.is_initialized = False
        
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize voice handler."""
        try:
            if PYQT_AVAILABLE:
                self.thread = VoiceHandlerThread()
                
                # Connect signals
                self.thread.transcription_ready.connect(self._on_transcription_ready)
                self.thread.error_occurred.connect(self._on_error)
                self.thread.recording_started.connect(lambda: self.recording_status_changed.emit(True))
                self.thread.recording_stopped.connect(lambda: self.recording_status_changed.emit(False))
                
                # Apply configuration if provided
                if config:
                    self._apply_config(config)
                
                logger.info("Voice handler initialized with PyQt threading")
            else:
                logger.warning("PyQt not available, falling back to standard threading")
                # TODO: Implement fallback threading version
                return False
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize voice handler: {e}")
            return False
    
    def _apply_config(self, config: Dict[str, Any]):
        """Apply configuration to voice handler."""
        if not self.thread:
            return
        
        # Audio config
        audio_config = config.get("audio", {})
        for key, value in audio_config.items():
            if hasattr(self.thread.audio_config, key):
                setattr(self.thread.audio_config, key, value)
        
        # Whisper config
        whisper_config = config.get("whisper", {})
        for key, value in whisper_config.items():
            if hasattr(self.thread.whisper_config, key):
                setattr(self.thread.whisper_config, key, value)
        
        # Recreate recorder with new config
        self.thread.recorder = VoiceRecorder(self.thread.audio_config)
    
    def start_listening(self, callback: Optional[Callable[[str, float], None]] = None) -> bool:
        """Start continuous voice recognition."""
        if not self.is_initialized or not self.thread:
            logger.error("Voice handler not initialized")
            return False
        
        if self.thread.isRunning():
            logger.warning("Voice handler already running")
            return False
        
        self.callback_function = callback
        self.thread.start()
        
        logger.info("Started voice recognition")
        return True
    
    def stop_listening(self):
        """Stop voice recognition."""
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(5000)  # Wait up to 5 seconds
            logger.info("Stopped voice recognition")
    
    def _on_transcription_ready(self, text: str, confidence: float):
        """Handle transcription ready signal."""
        self.transcription_received.emit(text, confidence)
        
        if self.callback_function:
            self.callback_function(text, confidence)
    
    def _on_error(self, error_message: str):
        """Handle error signal."""
        self.error_occurred.emit(error_message)
        logger.error(f"Voice handler error: {error_message}")
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_listening()
        logger.info("Voice handler cleaned up")

# Utility functions
def test_microphone() -> bool:
    """Test if microphone is available and working."""
    try:
        # Test recording for 1 second
        duration = 1.0
        sample_rate = 16000
        
        logger.info("Testing microphone...")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        
        # Check if we got audio data
        rms = np.sqrt(np.mean(recording**2))
        logger.info(f"Microphone test completed. RMS: {rms:.4f}")
        
        return rms > 0.001  # Basic threshold for detecting audio
        
    except Exception as e:
        logger.error(f"Microphone test failed: {e}")
        return False

def list_audio_devices() -> List[Dict[str, Any]]:
    """List available audio input devices."""
    try:
        devices = sd.query_devices()
        input_devices = []
        
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'index': idx,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })
        
        return input_devices
        
    except Exception as e:
        logger.error(f"Error listing audio devices: {e}")
        return [] 