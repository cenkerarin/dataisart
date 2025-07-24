# Voice Recognition Module

A comprehensive voice recognition system built with PyQt threading, sounddevice audio recording, and OpenAI Whisper transcription.

## Features

- **Non-blocking Operation**: Uses PyQt QThread for seamless GUI integration
- **Real-time Audio Recording**: Records audio using sounddevice with configurable parameters
- **Silence Detection**: Automatically stops recording after detecting silence
- **WAV File Segmentation**: Saves audio segments as temporary WAV files
- **Whisper Transcription**: Uses OpenAI's Whisper model for accurate speech-to-text
- **Error Handling**: Comprehensive error handling with retry mechanisms
- **Signal-based Communication**: PyQt signals for thread-safe communication

## Architecture

### Core Components

1. **AudioConfig**: Configuration dataclass for audio recording parameters
2. **WhisperConfig**: Configuration dataclass for Whisper model settings
3. **AudioSegment**: Represents recorded audio segments with file management
4. **VoiceRecorder**: Handles audio recording and silence detection
5. **VoiceHandlerThread**: PyQt thread for voice processing
6. **VoiceHandler**: Main interface class with signal integration

### Audio Processing Flow

```
Microphone Input → SoundDevice Recording → Silence Detection → 
Audio Segmentation → WAV File Creation → Whisper Transcription → 
Signal Emission → UI Update
```

## Installation

Install the required dependencies:

```bash
pip install sounddevice scipy openai-whisper PyQt5 numpy
```

## Usage

### Basic Usage

```python
from voice_handler import VoiceHandler

# Create voice handler
voice_handler = VoiceHandler()

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

# Initialize
success = voice_handler.initialize(config)

if success:
    # Start listening
    voice_handler.start_listening()
```

### PyQt Integration

```python
from PyQt5.QtCore import QObject
from voice_handler import VoiceHandler

class MyApp(QObject):
    def __init__(self):
        super().__init__()
        self.voice_handler = VoiceHandler(self)
        
        # Connect signals
        self.voice_handler.transcription_received.connect(self.on_transcription)
        self.voice_handler.error_occurred.connect(self.on_error)
        self.voice_handler.recording_status_changed.connect(self.on_recording_status)
        
        self.voice_handler.initialize()
    
    def on_transcription(self, text: str, confidence: float):
        print(f"Transcribed: {text} (Confidence: {confidence:.2f})")
    
    def on_error(self, error_message: str):
        print(f"Error: {error_message}")
    
    def on_recording_status(self, is_recording: bool):
        print(f"Recording: {'Yes' if is_recording else 'No'}")
```

### Callback Function Usage

```python
def transcription_callback(text: str, confidence: float):
    print(f"Received: {text} (Confidence: {confidence:.2f})")

voice_handler.start_listening(callback=transcription_callback)
```

## Configuration Options

### Audio Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_rate` | int | 16000 | Audio sample rate in Hz |
| `channels` | int | 1 | Number of audio channels (1=mono) |
| `dtype` | str | 'int16' | Audio data type |
| `chunk_duration` | float | 0.1 | Duration of each audio chunk in seconds |
| `silence_threshold` | float | 0.01 | RMS threshold for silence detection |
| `silence_duration` | float | 1.0 | Seconds of silence before stopping |
| `max_recording_duration` | float | 30.0 | Maximum recording length in seconds |
| `min_recording_duration` | float | 0.5 | Minimum recording length in seconds |

### Whisper Configuration  

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "base" | Whisper model size (tiny, base, small, medium, large) |
| `device` | str | "cpu" | Processing device ("cpu" or "cuda") |
| `language` | str | None | Language code (None for auto-detect) |
| `temperature` | float | 0.0 | Sampling temperature (0.0 = deterministic) |

## Available Whisper Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | 39 MB | Fastest | Lowest |
| base | 74 MB | Fast | Good |
| small | 244 MB | Medium | Better |
| medium | 769 MB | Slow | Very Good |
| large | 1550 MB | Slowest | Best |

## Signals

The `VoiceHandler` class emits the following PyQt signals:

- `transcription_received(str, float)`: Emitted when transcription is ready
- `error_occurred(str)`: Emitted when an error occurs
- `recording_status_changed(bool)`: Emitted when recording starts/stops

## Error Handling

The voice handler includes comprehensive error handling:

- **Microphone Access Errors**: Automatically retries with backoff
- **Recording Errors**: Graceful fallback and error reporting
- **Whisper Loading Errors**: Clear error messages and fallback options
- **File I/O Errors**: Automatic cleanup of temporary files

## Utility Functions

### Test Microphone

```python
from voice_handler import test_microphone

if test_microphone():
    print("Microphone is working!")
else:
    print("Microphone test failed!")
```

### List Audio Devices

```python
from voice_handler import list_audio_devices

devices = list_audio_devices()
for device in devices:
    print(f"Device: {device['name']} (Index: {device['index']})")
```

## Demo Application

Run the demo application to test the voice handler:

```bash
cd src/core/voice_recognition
python demo_voice.py
```

The demo provides:
- Start/Stop listening controls
- Real-time transcription display
- Microphone testing
- Audio device information
- Recording status indicator

## Troubleshooting

### Common Issues

1. **No audio devices found**
   - Check microphone permissions
   - Verify microphone is connected and working
   - Run `list_audio_devices()` to see available devices

2. **Whisper model loading fails**
   - Check internet connection (first download)
   - Verify sufficient disk space
   - Try smaller model (e.g., "tiny" instead of "base")

3. **Recording doesn't start**
   - Check microphone permissions
   - Verify audio device availability
   - Run microphone test first

4. **Poor transcription quality**
   - Adjust `silence_threshold` for your environment
   - Try larger Whisper model
   - Improve audio input quality

### Performance Tips

1. **Use appropriate Whisper model**:
   - `tiny` for fast, basic transcription
   - `base` for balanced performance
   - `small`+ for better accuracy

2. **Optimize audio settings**:
   - Higher sample rates for better quality
   - Adjust silence detection for your environment
   - Tune recording durations for your use case

3. **GPU acceleration**:
   - Set `device: "cuda"` if you have NVIDIA GPU
   - Significantly faster transcription

## File Structure

```
voice_recognition/
├── voice_handler.py      # Main voice handler implementation
├── demo_voice.py         # Demo application
└── README.md            # This file
```

## Dependencies

- **sounddevice**: Audio recording
- **scipy**: Audio file I/O
- **openai-whisper**: Speech transcription
- **PyQt5**: GUI threading and signals
- **numpy**: Audio data processing

## License

This module is part of the DataIsArt project. 