# 🤖 AI Assistant for Voice-Controlled Data Analysis

A comprehensive AI-powered assistant that integrates with voice recognition to provide hands-free data analysis and visualization capabilities.

## 🌟 Features

### Core Capabilities
- **Voice-Activated Commands**: Use natural language voice commands like "describe data" or "visualize sales column"
- **Smart Data Analysis**: Automatic dataset description, column analysis, and statistical summaries
- **Interactive Visualizations**: Generate charts, plots, and correlation matrices using Plotly
- **AI-Powered Parsing**: LLM-based command understanding with fallback to pattern matching
- **Real-time UI Updates**: Live display of results with suggestions for follow-up actions

### Supported Commands
- **Data Exploration**: "describe data", "show summary", "analyze [column]"
- **Visualization**: "visualize [column]", "create histogram", "scatter plot of X and Y", "correlation matrix"
- **Data Operations**: "filter data where [condition]", "sort by [column]"
- **Export**: "export data to [format]"

## 🛠️ Setup & Installation

### 1. Dependencies

Install required packages:

```bash
pip install pandas plotly openai PyQt5 QtWebEngine whisper sounddevice scipy numpy
```

### 2. API Configuration

Set up your OpenAI API key in `config/app/settings.py`:

```python
LLM_CONFIG = {
    "provider": "openai",
    "model": "gpt-3.5-turbo",  # or "gpt-4" for better results
    "api_key": "your-openai-api-key-here",  # Or set OPENAI_API_KEY environment variable
    "temperature": 0.1,
    "max_tokens": 1000,
    "fallback_to_patterns": True,  # Use pattern matching if API fails
}
```

### 3. Environment Variables (Recommended)

Instead of hardcoding the API key, set it as an environment variable:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

## 🚀 Quick Start

### Basic Usage

```python
from src.core.llm_integration import VoiceAIIntegration

# Configuration
config = {
    "llm": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "api_key": "your-key",  # or set as env var
        "temperature": 0.1,
        "max_tokens": 1000,
        "fallback_to_patterns": True
    },
    "voice": {
        "recognition_timeout": 5,
        "phrase_timeout": 1,
    }
}

# Initialize AI integration
ai_integration = VoiceAIIntegration(config)
ai_integration.initialize()

# Load dataset
result = ai_integration.load_dataset("data/your_dataset.csv")
print(f"Dataset loaded: {result.success}")

# Process text commands (for testing)
result = ai_integration.process_text_command("describe data")
print(f"Analysis result: {result.message}")

# Start voice recognition
ai_integration.start_voice_recognition()
```

### Demo Application

Run the complete demo application:

```python
python src/core/llm_integration/example_app.py
```

This launches a PyQt application with:
- Dataset loading interface
- Voice command controls
- Text command input (for testing)
- Real-time result display
- Interactive visualizations

## 🏗️ Architecture

### Core Components

```
AI Assistant System
├── AIAssistant (Core AI processing)
│   ├── DataAnalysisActions (Data operations)
│   ├── DatasetContext (Current dataset state)
│   └── ActionResult (Structured results)
├── VoiceAIBridge (Voice ↔ AI integration)
├── UIIntegration (PyQt display components)
└── Configuration Management
```

### Data Flow

```
Voice Input → Whisper Transcription → Command Parsing → AI Processing → Action Execution → UI Display
```

## 📝 Configuration Options

### LLM Settings

```python
LLM_CONFIG = {
    "provider": "openai",           # Currently only OpenAI supported
    "model": "gpt-3.5-turbo",      # Model to use
    "api_key": "",                 # API key (or env var)
    "org_id": "",                  # Optional organization ID
    "temperature": 0.1,            # Response randomness (0-1)
    "max_tokens": 1000,            # Maximum response length
    "timeout": 30,                 # Request timeout
    "fallback_to_patterns": True,  # Use pattern matching fallback
}
```

### Voice Recognition Settings

```python
VOICE_CONFIG = {
    "recognition_timeout": 5,       # Seconds to wait for speech
    "phrase_timeout": 1,           # Seconds of silence to end phrase
    "energy_threshold": 4000,      # Microphone sensitivity
    "dynamic_energy_threshold": True,
    "pause_threshold": 0.8,        # Pause detection
}
```

### Data Analysis Settings

```python
DATA_CONFIG = {
    "supported_formats": [".csv", ".xlsx", ".json", ".parquet"],
    "max_file_size_mb": 100,       # Maximum file size
    "sample_size_for_preview": 1000,
    "auto_detect_types": True,     # Auto-detect column types
    "cache_results": True,         # Cache analysis results
}
```

## 🎯 Usage Examples

### 1. Data Description

```python
# Voice: "Describe the data"
# Or programmatically:
result = ai_integration.process_text_command("describe data")

# Result includes:
# - Dataset shape and memory usage
# - Column information and types
# - Missing value analysis
# - Sample data preview
```

### 2. Visualization Creation

```python
# Voice: "Visualize age column"
# Voice: "Create scatter plot of income and age"
# Voice: "Show correlation matrix"

result = ai_integration.process_text_command("visualize sales")
# Creates interactive Plotly visualization
```

### 3. Column Analysis

```python
# Voice: "Analyze customer age column"
result = ai_integration.process_text_command("analyze price")

# Result includes:
# - Statistical summary (mean, median, std)
# - Value distribution
# - Missing value count
# - Data type information
```

## 🔧 Integration with Existing UI

### Adding to PyQt Application

```python
from src.core.llm_integration import AIResultDisplayWidget, VoiceAIIntegration

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize AI integration
        self.ai_integration = VoiceAIIntegration(config)
        self.ai_integration.initialize()
        
        # Create result display widget
        self.results_widget = AIResultDisplayWidget()
        self.ai_integration.set_ui_display(self.results_widget)
        
        # Add to your layout
        layout.addWidget(self.results_widget)
        
        # Connect signals
        self.results_widget.action_requested.connect(
            self.ai_integration.process_text_command
        )
```

### Voice Panel Integration

```python
# In your voice panel
from src.core.llm_integration import AICommandProcessor

class VoicePanel(QWidget):
    def __init__(self):
        super().__init__()
        
        # Connect with AI processor
        self.ai_processor = AICommandProcessor(ai_assistant)
        self.command_parsed.connect(self.ai_processor.handle_voice_command)
```

## 🐛 Troubleshooting

### Common Issues

1. **No API Key Error**
   ```
   Error: No API key provided
   Solution: Set OPENAI_API_KEY environment variable or configure in settings
   ```

2. **Voice Recognition Not Working**
   ```
   Error: Voice recognition not available
   Solution: Install dependencies: pip install whisper sounddevice
   ```

3. **Visualization Display Issues**
   ```
   Error: QtWebEngine not found
   Solution: Install PyQtWebEngine: pip install PyQtWebEngine
   ```

4. **Dataset Loading Errors**
   ```
   Error: Unsupported file format
   Solution: Use supported formats (.csv, .xlsx, .json) or extend DataAnalysisActions
   ```

### Performance Tips

1. **Large Datasets**: The system automatically uses sampling for previews
2. **API Costs**: Set `fallback_to_patterns: True` to reduce API calls
3. **Voice Latency**: Adjust `phrase_timeout` for faster/slower speech
4. **Memory Usage**: Monitor dataset size with `max_file_size_mb` setting

## 🔒 Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **Environment Variables**: Use `.env` files or system environment variables
3. **Data Privacy**: Consider data sensitivity when using cloud LLM APIs
4. **Local Processing**: Voice transcription happens locally with Whisper

## 🧪 Testing

### Manual Testing

```python
# Test without voice
ai_integration = VoiceAIIntegration(config)
ai_integration.initialize()

# Test commands
commands = [
    "describe data",
    "visualize first_column", 
    "analyze second_column",
    "show correlation matrix"
]

for cmd in commands:
    result = ai_integration.process_text_command(cmd)
    print(f"{cmd}: {result.success}")
```

### Demo Mode

Run without API key to test pattern-matching fallback:

```python
config = {
    "llm": {"api_key": "", "fallback_to_patterns": True},
    "voice": {}
}
```

## 🚀 Advanced Usage

### Custom Actions

Extend the AI assistant with custom data analysis actions:

```python
class CustomDataActions(DataAnalysisActions):
    def custom_analysis(self, **kwargs) -> ActionResult:
        # Your custom analysis logic
        return ActionResult(
            success=True,
            action_type=ActionType.ANALYZE_COLUMN,
            message="Custom analysis complete"
        )

# Register custom actions
ai_assistant.action_handlers[ActionType.CUSTOM] = custom_actions.custom_analysis
```

### Custom Voice Commands

Add new command patterns to the voice parser:

```python
# In command_parser.py, add new patterns
custom_patterns = {
    CommandIntent.CUSTOM: [
        (r"(custom analysis|special report)", {"extract": []}),
    ]
}
```

## 📊 Example Results

When you say "describe data" for a sales dataset:

```
✅ Describe Data
Dataset 'sales_data' has 1000 rows and 8 columns.
Memory usage: 24.5 MB.
Columns with missing values: customer_notes.

Data Analysis:
=== DATASET OVERVIEW ===
Name: sales_data
Shape: (1000, 8)
Memory: 24.5 MB

=== COLUMNS ===
product_id: Type: int64, Non-null: 1000, Unique: 150
price: Type: float64, Non-null: 1000, Unique: 89
customer_age: Type: int64, Non-null: 998, Unique: 45

💡 Suggestions:
• Try 'visualize column distribution' to see data patterns
• Use 'show correlation matrix' to find relationships
• Say 'filter data where [condition]' to subset the data
```

## 🤝 Contributing

1. **Adding New Actions**: Extend `DataAnalysisActions` class
2. **UI Components**: Create new display widgets in `ui_integration.py`
3. **Voice Commands**: Add patterns to `command_parser.py`
4. **Visualizations**: Extend visualization options in `visualize_data()` method

## 📄 License

This project is part of the Hands-Free Data Science application. 