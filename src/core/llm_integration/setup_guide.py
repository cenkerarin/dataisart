"""
AI Assistant Setup Guide
========================

Interactive setup guide for configuring the AI assistant system.
Helps users set up API keys, test components, and verify installation.
"""

import os
import sys
from pathlib import Path
import json

def print_header():
    """Print welcome header."""
    print("=" * 60)
    print("🤖 AI Assistant Setup Guide")
    print("=" * 60)
    print("Welcome! This guide will help you set up the AI assistant system.")
    print()

def check_dependencies():
    """Check if required dependencies are installed."""
    print("📦 Checking Dependencies...")
    print("-" * 30)
    
    required_packages = {
        'pandas': 'Data manipulation',
        'plotly': 'Interactive visualizations', 
        'openai': 'OpenAI API client',
        'PyQt5': 'GUI framework',
        'whisper': 'Voice transcription',
        'sounddevice': 'Audio recording',
        'scipy': 'Scientific computing',
        'numpy': 'Numerical computing'
    }
    
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"❌ {package} - {description} (MISSING)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n🔧 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✅ All dependencies are installed!")
    return True

def setup_api_key():
    """Setup OpenAI API key."""
    print("\n🔑 API Key Setup")
    print("-" * 20)
    
    # Check if already set
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✅ OPENAI_API_KEY environment variable is already set")
        print(f"   Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else ''}")
        
        keep = input("Keep current API key? (y/n): ").lower().strip()
        if keep == 'y':
            return api_key
    
    print("\nOption 1: Set as environment variable (Recommended)")
    print("Option 2: Enter directly (for testing only)")
    print("Option 3: Skip (use demo mode)")
    
    choice = input("\nChoose option (1/2/3): ").strip()
    
    if choice == '1':
        print("\nAdd this to your shell profile (.bashrc, .zshrc, etc.):")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr run this command in your terminal:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        
        api_key = input("\nEnter your API key to test: ").strip()
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            return api_key
            
    elif choice == '2':
        api_key = input("Enter your OpenAI API key: ").strip()
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            return api_key
            
    elif choice == '3':
        print("⚠️  Skipping API key setup. AI assistant will run in demo mode.")
        print("   (Pattern matching only, no LLM processing)")
        return ""
    
    return ""

def test_api_connection(api_key):
    """Test OpenAI API connection."""
    if not api_key:
        print("🔄 Skipping API test (no key provided)")
        return True
    
    print("\n🔄 Testing OpenAI API Connection...")
    print("-" * 35)
    
    try:
        import openai
        
        client = openai.OpenAI(api_key=api_key)
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'API test successful'"}],
            max_tokens=10
        )
        
        result = response.choices[0].message.content
        print(f"✅ API connection successful!")
        print(f"   Response: {result}")
        return True
        
    except Exception as e:
        print(f"❌ API connection failed: {str(e)}")
        print("   Please check your API key and internet connection")
        return False

def test_voice_components():
    """Test voice recognition components."""
    print("\n🎤 Testing Voice Components...")
    print("-" * 32)
    
    try:
        import whisper
        print("✅ Whisper (voice transcription) is available")
    except ImportError:
        print("❌ Whisper not found. Install with: pip install whisper")
        return False
    
    try:
        import sounddevice as sd
        
        # Test microphone access
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        if input_devices:
            print(f"✅ Found {len(input_devices)} input device(s)")
            for i, device in enumerate(input_devices[:3]):  # Show first 3
                print(f"   {i+1}. {device['name']}")
        else:
            print("⚠️  No input devices found")
            return False
            
    except ImportError:
        print("❌ sounddevice not found. Install with: pip install sounddevice")
        return False
    except Exception as e:
        print(f"⚠️  Audio device check failed: {str(e)}")
    
    print("✅ Voice components are ready!")
    return True

def create_config_file(api_key):
    """Create configuration file."""
    print("\n📝 Creating Configuration...")
    print("-" * 30)
    
    config = {
        "llm": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "api_key": api_key,
            "temperature": 0.1,
            "max_tokens": 1000,
            "fallback_to_patterns": True
        },
        "voice": {
            "recognition_timeout": 5,
            "phrase_timeout": 1,
            "energy_threshold": 4000,
            "dynamic_energy_threshold": True,
            "pause_threshold": 0.8
        },
        "data": {
            "supported_formats": [".csv", ".xlsx", ".json", ".parquet"],
            "max_file_size_mb": 100,
            "sample_size_for_preview": 1000,
            "auto_detect_types": True,
            "cache_results": True
        },
        "visualization": {
            "default_theme": "plotly_dark",
            "figure_width": 800,
            "figure_height": 600,
            "export_formats": ["png", "pdf", "svg", "html"],
            "interactive": True
        }
    }
    
    # Save to config directory
    config_dir = Path(__file__).parent.parent.parent.parent / "config"
    config_file = config_dir / "ai_assistant_config.json"
    
    try:
        config_dir.mkdir(exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(config, indent=2, fp=f)
        
        print(f"✅ Configuration saved to: {config_file}")
        return str(config_file)
        
    except Exception as e:
        print(f"⚠️  Could not save config file: {str(e)}")
        print("You can create the configuration manually in your code.")
        return None

def test_ai_assistant():
    """Test the AI assistant components."""
    print("\n🤖 Testing AI Assistant...")
    print("-" * 28)
    
    try:
        # Add project path
        project_root = Path(__file__).parent.parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        from src.core.llm_integration import VoiceAIIntegration
        
        # Basic configuration
        config = {
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv('OPENAI_API_KEY', ''),
                "fallback_to_patterns": True
            },
            "voice": {},
            "data": {}
        }
        
        # Initialize AI integration
        ai_integration = VoiceAIIntegration(config)
        success = ai_integration.initialize()
        
        if success:
            print("✅ AI Assistant initialized successfully")
            
            # Test basic command
            result = ai_integration.process_text_command("describe data")
            if result.success:
                print("✅ Command processing works")
            else:
                print(f"⚠️  Command processing: {result.error}")
            
            return True
        else:
            print("⚠️  AI Assistant initialization had issues (but may still work)")
            return True
            
    except Exception as e:
        print(f"❌ AI Assistant test failed: {str(e)}")
        return False

def run_demo():
    """Offer to run demo application."""
    print("\n🚀 Demo Application")
    print("-" * 20)
    
    choice = input("Would you like to run the demo application? (y/n): ").lower().strip()
    
    if choice == 'y':
        try:
            demo_path = Path(__file__).parent / "example_app.py"
            print(f"\nStarting demo application...")
            print(f"File: {demo_path}")
            print("\nTip: Load a CSV file and try commands like:")
            print("• 'describe data'")
            print("• 'visualize [column_name]'")
            print("• 'analyze [column_name]'")
            
            # Import and run
            import subprocess
            subprocess.run([sys.executable, str(demo_path)])
            
        except Exception as e:
            print(f"❌ Could not run demo: {str(e)}")
            print(f"\nYou can run it manually with:")
            print(f"python {demo_path}")

def print_next_steps():
    """Print next steps and usage information."""
    print("\n🎯 Next Steps")
    print("-" * 15)
    print("1. Load a dataset (CSV, Excel, or JSON)")
    print("2. Try voice commands:")
    print("   • 'Describe data' - Get dataset overview")
    print("   • 'Visualize [column]' - Create charts")
    print("   • 'Analyze [column]' - Detailed column analysis")
    print("   • 'Show correlation matrix' - See relationships")
    print()
    print("3. Use text commands for testing without voice")
    print()
    print("📚 Documentation:")
    print("   • README.md - Complete setup guide")
    print("   • example_app.py - Full demo application")
    print("   • Voice commands list in README.md")
    print()
    print("🐛 Troubleshooting:")
    print("   • Check dependencies with this script")
    print("   • Set OPENAI_API_KEY environment variable")
    print("   • Use fallback mode if API issues occur")

def main():
    """Main setup function."""
    print_header()
    
    # Step 1: Check dependencies
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n❌ Please install missing dependencies first.")
        return
    
    # Step 2: Setup API key
    api_key = setup_api_key()
    
    # Step 3: Test API if key provided
    if api_key:
        test_api_connection(api_key)
    
    # Step 4: Test voice components
    test_voice_components()
    
    # Step 5: Create config file
    create_config_file(api_key)
    
    # Step 6: Test AI assistant
    test_ai_assistant()
    
    # Step 7: Offer demo
    run_demo()
    
    # Step 8: Next steps
    print_next_steps()
    
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main() 