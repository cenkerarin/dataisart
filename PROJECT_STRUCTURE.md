# 📁 Project Structure Documentation

## Overview
This document provides a comprehensive guide to the **Hands-Free Data Science** project structure, explaining the purpose and organization of each directory and file.

## 📂 Directory Structure

```
dataisart/
├── 📋 README.md                          # Main project documentation
├── 📄 PROJECT_STRUCTURE.md               # This file - project structure guide
├── 🚀 main.py                           # Main application entry point
├── 📦 requirements.txt                   # Python dependencies
├── 
├── 📁 src/                              # Source code directory
│   ├── 📁 core/                         # Core functionality modules
│   │   ├── 📁 hand_tracking/            # Hand gesture recognition
│   │   │   ├── 🖐️ gesture_detector.py   # MediaPipe hand tracking
│   │   │   ├── 📊 gesture_classifier.py # Gesture classification logic
│   │   │   └── 🎯 selection_handler.py  # Data selection from gestures
│   │   │
│   │   ├── 📁 voice_recognition/        # Voice command processing
│   │   │   ├── 🎤 voice_handler.py      # Speech-to-text conversion
│   │   │   ├── 🧠 command_parser.py     # Command intent recognition
│   │   │   └── 🎙️ audio_processor.py    # Audio preprocessing
│   │   │
│   │   ├── 📁 llm_integration/          # AI assistant functionality
│   │   │   ├── 🤖 ai_assistant.py       # OpenAI GPT integration
│   │   │   ├── 💬 prompt_manager.py     # Prompt engineering
│   │   │   └── 🔄 response_parser.py    # AI response processing
│   │   │
│   │   ├── 📁 data_processing/          # Data manipulation
│   │   │   ├── 📊 data_manager.py       # Dataset loading & management
│   │   │   ├── 🔍 data_analyzer.py      # Statistical analysis
│   │   │   └── 🧹 data_cleaner.py       # Data preprocessing
│   │   │
│   │   └── 📁 visualization/            # Chart generation
│   │       ├── 📈 chart_generator.py    # Plotly visualizations
│   │       ├── 🎨 style_manager.py      # Chart styling
│   │       └── 📊 plot_templates.py     # Pre-built chart templates
│   │
│   ├── 📁 ui/                           # User interface components
│   │   ├── 📁 pages/                    # Streamlit pages
│   │   │   ├── 🏠 main_dashboard.py     # Main application dashboard
│   │   │   ├── 📊 data_explorer.py      # Data exploration page
│   │   │   ├── 🎯 gesture_training.py   # Gesture training interface
│   │   │   └── ⚙️ settings.py           # Application settings
│   │   │
│   │   ├── 📁 components/               # Reusable UI components
│   │   │   ├── 📹 camera_widget.py      # Camera display widget
│   │   │   ├── 🎤 voice_widget.py       # Voice input widget
│   │   │   ├── 📊 data_table.py         # Enhanced data table
│   │   │   └── 📈 chart_widget.py       # Chart display widget
│   │   │
│   │   └── 📁 styles/                   # UI styling
│   │       ├── 🎨 app_styles.css        # Main application styles
│   │       └── 📱 mobile_styles.css     # Mobile-responsive styles
│   │
│   └── 📁 utils/                        # Utility functions
│       ├── 📁 helpers/                  # Helper functions
│       │   ├── ⚙️ config_loader.py      # Configuration management
│       │   ├── 📝 logger.py             # Logging utilities
│       │   └── 🔧 file_utils.py         # File handling utilities
│       │
│       └── 📁 validators/               # Input validation
│           ├── 📊 data_validators.py    # Dataset validation
│           ├── 🎤 voice_validators.py   # Voice command validation
│           └── 🖐️ gesture_validators.py # Gesture validation
│
├── 📁 config/                           # Configuration files
│   ├── 📁 app/                          # Application configuration
│   │   ├── ⚙️ settings.py               # Main app settings
│   │   └── 🌍 environment.py            # Environment variables
│   │
│   ├── 📁 models/                       # Model configurations
│   │   ├── 🤖 llm_config.py             # LLM model settings
│   │   ├── 🖐️ gesture_config.py        # Hand tracking settings
│   │   └── 🎤 voice_config.py           # Voice recognition settings
│   │
│   └── 📁 api/                          # API configurations
│       ├── 🔑 openai_config.py          # OpenAI API settings
│       └── 🌐 endpoints.py              # API endpoints
│
├── 📁 data/                             # Data storage
│   ├── 📁 raw/                          # Raw uploaded datasets
│   ├── 📁 processed/                    # Processed datasets
│   └── 📁 sample/                       # Sample datasets for testing
│       ├── 🌺 iris.csv                  # Iris dataset
│       ├── 🚢 titanic.csv               # Titanic dataset
│       └── 🏠 housing.csv               # Boston housing dataset
│
├── 📁 tests/                            # Test files
│   ├── 📁 unit/                         # Unit tests
│   │   ├── 🧪 test_data_manager.py      # Data manager tests
│   │   ├── 🧪 test_gesture_detector.py  # Gesture detection tests
│   │   ├── 🧪 test_voice_handler.py     # Voice recognition tests
│   │   └── 🧪 test_ai_assistant.py      # AI assistant tests
│   │
│   ├── 📁 integration/                  # Integration tests
│   │   ├── 🔗 test_voice_to_ai.py       # Voice-to-AI integration
│   │   ├── 🔗 test_gesture_to_data.py   # Gesture-to-data integration
│   │   └── 🔗 test_full_pipeline.py     # End-to-end tests
│   │
│   ├── 📁 core/                         # Core module tests
│   ├── 📁 ui/                           # UI component tests
│   └── 📁 utils/                        # Utility function tests
│
├── 📁 docs/                             # Documentation
│   ├── 📁 api/                          # API documentation
│   │   ├── 📖 core_modules.md           # Core module APIs
│   │   └── 📖 ui_components.md          # UI component APIs
│   │
│   ├── 📁 user_guide/                   # User documentation
│   │   ├── 📘 getting_started.md        # Getting started guide
│   │   ├── 📘 voice_commands.md         # Voice command reference
│   │   ├── 📘 hand_gestures.md          # Hand gesture guide
│   │   └── 📘 troubleshooting.md        # Troubleshooting guide
│   │
│   └── 📁 development/                  # Developer documentation
│       ├── 🔧 setup.md                  # Development setup
│       ├── 🔧 contributing.md           # Contribution guidelines
│       └── 🔧 architecture.md           # System architecture
│
├── 📁 assets/                           # Static assets
│   ├── 📁 images/                       # Images and screenshots
│   │   ├── 🖼️ logo.png                  # Application logo
│   │   ├── 🖼️ demo_screenshot.png       # Demo screenshots
│   │   └── 🖼️ gesture_examples.png      # Gesture examples
│   │
│   ├── 📁 videos/                       # Demo videos
│   │   ├── 🎥 demo_video.mp4            # Application demo
│   │   └── 🎥 gesture_tutorial.mp4      # Gesture tutorial
│   │
│   ├── 📁 audio/                        # Audio files
│   │   ├── 🔊 notification.wav          # UI notifications
│   │   └── 🔊 voice_samples.wav         # Voice command samples
│   │
│   ├── 📁 icons/                        # UI icons
│   ├── 📁 fonts/                        # Custom fonts
│   └── 📁 sample_data/                  # Sample datasets
│
├── 📁 logs/                             # Application logs
│   ├── 📊 app.log                       # Main application log
│   ├── 🖐️ gesture.log                   # Gesture detection log
│   ├── 🎤 voice.log                     # Voice recognition log
│   └── 🤖 ai.log                        # AI assistant log
│
└── 📁 notebooks/                        # Jupyter notebooks
    ├── 📓 data_exploration.ipynb         # Data exploration examples
    ├── 📓 gesture_analysis.ipynb         # Gesture detection analysis
    ├── 📓 voice_testing.ipynb            # Voice recognition testing
    └── 📓 ai_prompt_testing.ipynb        # AI prompt engineering
```

## 🚀 Quick Start Guide

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd dataisart

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.template .env
# Edit .env with your OpenAI API key
```

### 2. Configuration
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Optional: Set other environment variables
export DEBUG=True
export LOG_LEVEL=INFO
```

### 3. Run the Application
```bash
# Start the Streamlit application
streamlit run main.py

# Or run directly with Python
python main.py
```

## 📋 Core Components

### 🖐️ Hand Tracking (`src/core/hand_tracking/`)
- **Purpose**: Detect and interpret hand gestures using MediaPipe
- **Key Features**:
  - Real-time hand landmark detection
  - Gesture classification (pointing, selection, etc.)
  - Data region selection through hand movements
  - Integration with dataset viewer

### 🎤 Voice Recognition (`src/core/voice_recognition/`)
- **Purpose**: Convert speech to text and parse commands
- **Key Features**:
  - Multiple speech recognition engines (Google, Whisper)
  - Command intent classification
  - Continuous listening mode
  - Voice command validation

### 🤖 LLM Integration (`src/core/llm_integration/`)
- **Purpose**: AI-powered data analysis and visualization
- **Key Features**:
  - OpenAI GPT-4 integration
  - Context-aware prompts with dataset information
  - Structured response parsing
  - Conversation history management

### 📊 Data Processing (`src/core/data_processing/`)
- **Purpose**: Dataset management and analysis
- **Key Features**:
  - Multiple format support (CSV, Excel, JSON, Parquet)
  - Statistical analysis and summaries
  - Data selection and filtering
  - Memory-efficient processing

### 📈 Visualization (`src/core/visualization/`)
- **Purpose**: Generate interactive charts and plots
- **Key Features**:
  - Plotly-based interactive visualizations
  - Multiple chart types (histogram, scatter, bar, line)
  - Dynamic chart generation from voice commands
  - Customizable styling and themes

## 🎯 Usage Examples

### Voice Commands
```
"Show me the age distribution"
"Visualize the correlation between age and income"
"Create a scatter plot of height vs weight"
"Analyze the missing values in the dataset"
"Filter the data where age is greater than 30"
```

### Hand Gestures
- **Pointing**: Select individual data points or columns
- **Box Selection**: Draw selection boxes around data regions
- **Swipe**: Navigate through dataset pages
- **Pinch**: Zoom in/out of visualizations

## 🔧 Development Guidelines

### Adding New Features
1. **Core Modules**: Add new functionality to `src/core/`
2. **UI Components**: Create reusable components in `src/ui/components/`
3. **Tests**: Write tests in appropriate `tests/` subdirectories
4. **Documentation**: Update relevant documentation in `docs/`

### Code Organization
- Follow the existing directory structure
- Use type hints for better code clarity
- Include docstrings for all public methods
- Follow PEP 8 style guidelines

### Testing
```bash
# Run unit tests
python -m pytest tests/unit/

# Run integration tests
python -m pytest tests/integration/

# Run all tests
python -m pytest tests/
```

## 📝 Configuration Management

### Environment Variables
- `OPENAI_API_KEY`: Required for AI functionality
- `DEBUG`: Enable debug mode
- `LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `CAMERA_DEVICE_ID`: Camera device identifier
- `VOICE_RECOGNITION_TIMEOUT`: Voice recognition timeout

### Application Settings
- Modify `config/app/settings.py` for application-wide settings
- Adjust `config/models/` for model-specific configurations
- Update `config/api/` for API-related settings

## 🚨 Troubleshooting

### Common Issues
1. **Camera not working**: Check camera permissions and device ID
2. **Voice recognition failing**: Verify microphone permissions and audio levels
3. **AI not responding**: Ensure OpenAI API key is correctly set
4. **Import errors**: Check that all dependencies are installed

### Debug Mode
Enable debug mode by setting `DEBUG=True` in your environment variables for detailed logging and error messages.

## 📚 Additional Resources

- **User Guide**: `docs/user_guide/getting_started.md`
- **API Documentation**: `docs/api/core_modules.md`
- **Development Setup**: `docs/development/setup.md`
- **Contributing Guidelines**: `docs/development/contributing.md`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

---

*This project structure is designed to be modular, scalable, and maintainable. Each component has a specific responsibility, making it easy to extend and modify the application.* 