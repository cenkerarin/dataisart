# DataIsArt Project Structure

This document outlines the structure of the hands-free data science application.

## Core Structure

```
dataisart/
├── config/
│   └── app/
│       ├── environment.py          # Environment configuration
│       └── settings.py             # Application settings and LLM config
├── src/
│   ├── core/
│   │   ├── data_processing/        # Data management utilities
│   │   ├── hand_tracking/          # Hand gesture recognition system
│   │   ├── llm_integration/        # AI assistant integration
│   │   │   ├── ai_assistant.py     # Core AI assistant with data analysis
│   │   │   ├── ui_integration.py   # PyQt UI components for AI results
│   │   │   ├── voice_ai_bridge.py  # Voice-AI integration bridge
│   │   │   └── __init__.py         # Module exports
│   │   └── voice_recognition/      # Voice command processing
│   ├── ui/                         # User interface components
│   │   ├── main_window.py          # Main application window
│   │   └── widgets/                # UI widgets
│   └── utils/                      # Utility functions
├── main.py                         # Application entry point
├── run_app.py                      # Alternative app runner
└── requirements.txt                # Python dependencies
```

## Key Components

### LLM Integration (`src/core/llm_integration/`)

**AIAssistant (`ai_assistant.py`)**
- Core AI assistant with OpenAI integration
- Data analysis actions (describe, visualize, analyze columns)
- Dataset context management
- Fallback to pattern matching when API unavailable

**Voice-AI Bridge (`voice_ai_bridge.py`)**
- Connects voice recognition to AI processing
- Handles asynchronous command processing
- Manages voice activation and status

**UI Integration (`ui_integration.py`)**
- PyQt widgets for displaying AI results
- Interactive visualization display
- Command suggestion system

### Voice Recognition (`src/core/voice_recognition/`)
- Voice command capture and transcription
- Command parsing and intent recognition
- Integration with gesture-based activation

### Hand Tracking (`src/core/hand_tracking/`)
- Gesture detection and classification
- Training tools and data collection
- Integration with voice activation

### Configuration (`config/app/`)
- LLM settings (API keys, model configuration)
- Voice recognition parameters
- Data processing settings
- Visualization preferences

## Usage

The system integrates voice recognition with AI-powered data analysis:

1. **Voice Command**: User speaks data analysis request
2. **AI Processing**: OpenAI API processes natural language command
3. **Action Execution**: System performs data analysis or visualization
4. **UI Display**: Results shown with interactive elements

### Core Features

- **Voice-activated data analysis**
- **Natural language command processing**
- **Interactive visualizations**
- **Smart fallback to pattern matching**
- **Real-time UI updates**

### Configuration

Set up in `config/app/settings.py`:
- OpenAI API key and model settings
- Voice recognition parameters
- Data processing options
- Visualization preferences 