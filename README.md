# 🧠🖐️ Hands-Free Data Science

An innovative application that combines **hand gesture recognition** and **voice commands** with **AI-powered data analysis** for hands-free data science workflows.

## ✨ Features

- **Voice-Activated AI Assistant**: Natural language commands for data analysis
- **Hand Gesture Recognition**: Control workflows with hand gestures
- **Interactive Visualizations**: AI-generated charts and graphs
- **Smart Command Processing**: OpenAI-powered with pattern matching fallback
- **Real-time UI**: Live display of analysis results and suggestions

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   cp env.template .env
   # Edit .env with your actual API key
   ```

2. **Run Application**
   ```bash
   python main.py
   ```

3. **Use Voice Commands**
   - Load a dataset
   - Say: "describe data", "visualize sales", "analyze age column"
   - Watch AI-powered analysis in real-time

## 🏗️ Core Components

- **AI Assistant**: OpenAI-powered natural language processing
- **Voice Recognition**: Speech-to-text with command parsing
- **Hand Tracking**: MediaPipe-based gesture recognition
- **UI Integration**: PyQt interface with interactive results
- **Data Processing**: Automated analysis and visualization

## 📊 Voice Commands

- **Data Analysis**: "describe data", "analyze [column]"
- **Visualization**: "visualize [column]", "correlation matrix"
- **Natural Language**: "tell me about this dataset", "what insights do you see?"

## ⚙️ Configuration

1. **Create environment file**:
   ```bash
   cp env.template .env
   # Edit .env with your actual API key
   ```

2. **Secure Setup**: All sensitive data uses environment variables - never commit secrets!

## 🔒 Security

Before pushing to GitHub, run the security check:
```bash
python3 check_security.py
```

See `SECURITY.md` for detailed security guidelines.

## 📄 License

Educational and research use.
