# Voice Recognition Integration Guide

## Overview

This guide explains how voice recognition has been integrated into the UI with gesture-based activation. The system allows users to activate voice commands by showing a **peace sign (✌️)** gesture to the camera.

## Architecture

### Components

1. **VoicePanel** (`src/ui/widgets/voice_panel.py`)
   - Main voice recognition interface
   - Handles gesture-triggered activation
   - Parses voice commands using the command parser
   - Displays real-time status and command history

2. **CommandParser** (`src/core/voice_recognition/command_parser.py`)
   - Parses natural language voice commands
   - Supports data analysis operations (filter, sort, visualize, etc.)
   - Returns structured command data

3. **VoiceHandler** (`src/core/voice_recognition/voice_handler.py`)
   - Handles audio recording and transcription
   - Uses OpenAI Whisper for speech-to-text
   - Manages audio processing pipeline

4. **MainWindow Integration** (`src/ui/main_window.py`)
   - Integrates voice panel into 3-panel layout
   - Connects gesture detection to voice activation
   - Routes voice commands to data panel

## How It Works

### 1. Gesture-Based Activation

- **Trigger Gesture**: Peace sign (✌️)
- **Hold Duration**: 1 second
- **Cooldown**: 3 seconds between activations
- **Confidence Threshold**: 70%

```python
# Voice panel monitors gestures from camera
def check_gesture_activation(self):
    if self.current_gesture == "peace_sign":
        if hold_duration >= self.gesture_hold_duration:
            self.activate_voice_recognition()
```

### 2. Voice Command Flow

```
Camera → Gesture Detection → Peace Sign → Voice Activation → 
Audio Recording → Whisper Transcription → Command Parsing → 
Data Panel Action
```

### 3. Supported Commands

The system recognizes these voice command patterns:

#### Visualization Commands
- "visualize column age"
- "create bar chart for sales"
- "histogram of prices"
- "plot revenue"

#### Filter Commands
- "filter country equals Germany"
- "where age greater than 25"
- "show only status active"
- "filter name contains John"

#### Sort Commands
- "sort by score descending"
- "order by age ascending"
- "sort name descending"

#### Aggregation Commands
- "sum of revenue"
- "average age"
- "count records"
- "max of score"

#### Data Commands
- "show data"
- "describe dataset"
- "display table"

## UI Layout

```
┌─────────────────┬─────────────────┬─────────────────┐
│   Data Panel    │  Camera Panel   │  Voice Panel    │
│                 │                 │                 │
│ • Dataset view  │ • Live camera   │ • Voice status  │
│ • Gesture ctrl  │ • Hand tracking │ • Gesture info  │
│ • Voice actions │ • Peace sign ✌️ │ • Manual ctrl   │
│                 │   detection     │ • Command hist  │
└─────────────────┴─────────────────┴─────────────────┘
```

## Usage Instructions

### 1. Start the Application
```bash
cd src/ui
python test_voice_integration.py
```

### 2. Enable Components
1. Click "📹 Start Camera" in camera panel
2. Ensure "Enable Voice Recognition" is checked in voice panel
3. Test microphone with "🔊 Test Microphone" button

### 3. Use Gesture Activation
1. Show peace sign (✌️) to camera
2. Hold for 1 second (progress bar will appear)
3. Speak your command clearly
4. Wait for processing and results

### 4. Manual Activation
- Use "🎤 Start Voice Recognition" button for manual control
- Useful for testing without gesture detection

## Configuration

### Voice Settings
```python
config = {
    "audio": {
        "silence_duration": 1.5,      # Stop after 1.5s silence
        "max_recording_duration": 10.0,  # Max 10 seconds
        "silence_threshold": 0.01,    # Sensitivity
        "min_recording_duration": 0.5   # Minimum recording
    },
    "whisper": {
        "model_name": "base",         # Whisper model
        "temperature": 0.0            # Deterministic output
    }
}
```

### Gesture Settings
```python
# In VoicePanel.__init__()
self.trigger_gesture = "peace_sign"      # Trigger gesture
self.gesture_hold_duration = 1.0         # Hold time (seconds)
self.gesture_cooldown = 3.0              # Cooldown (seconds)
```

## Troubleshooting

### Common Issues

1. **Voice Recognition Not Working**
   - Check microphone permissions
   - Ensure ffmpeg is installed: `brew install ffmpeg`
   - Test microphone with test button

2. **Gesture Not Detected**
   - Ensure camera is started
   - Check lighting conditions
   - Show clear peace sign ✌️ gesture
   - Verify gesture classification is enabled

3. **Commands Not Recognized**
   - Speak clearly and slowly
   - Use supported command patterns
   - Check command history for transcription accuracy

4. **Performance Issues**
   - Close other applications using microphone/camera
   - Reduce Whisper model size if needed
   - Check system resources

### Debug Information

- **Voice Panel**: Shows real-time status and command history
- **Status Bar**: Displays voice recognition status
- **Console**: Check for error messages and debug output

## Dependencies

### Required
- PyQt5
- sounddevice
- scipy
- openai-whisper
- ffmpeg (system dependency)

### Optional
- spacy (for advanced NLP parsing)

## Future Enhancements

1. **More Gestures**: Add support for additional trigger gestures
2. **Voice Feedback**: Audio confirmation of commands
3. **Command Customization**: User-defined voice commands
4. **Multi-language**: Support for different languages
5. **Continuous Listening**: Always-on voice recognition mode
6. **Command Shortcuts**: Quick voice shortcuts for common actions

## API Reference

### VoicePanel Methods

```python
# Update gesture status from camera
voice_panel.update_gesture_status(gesture_name, confidence)

# Manual voice control
voice_panel.toggle_manual_listening()

# Test microphone
voice_panel.test_microphone()
```

### Signal Connections

```python
# Gesture to voice activation
camera_panel.gesture_detected.connect(main_window.handle_gesture_for_voice)

# Voice commands to data actions
voice_panel.command_parsed.connect(data_panel.handle_voice_command)

# Status updates
voice_panel.voice_status_changed.connect(status_bar.update_voice_status)
```

## Example Commands

Try these example commands after showing the peace sign:

- "describe data" - Show dataset information
- "filter age greater than 30" - Filter data
- "sort by name ascending" - Sort data
- "visualize column sales" - Create visualization
- "sum of revenue" - Calculate aggregation
- "show data" - Display dataset

The system will show the parsed command in the voice panel and execute the corresponding action in the data panel. 