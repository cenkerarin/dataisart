#!/usr/bin/env python3
"""
Voice Recognition Pipeline Test
===============================

Command-line test to verify that voice recognition and command parsing
are working correctly. This helps isolate issues without the GUI.
"""

import sys
import time
import logging
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set up logging to see detailed information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

try:
    from voice_handler import VoiceHandler, test_microphone, list_audio_devices
    from command_parser import create_parser, CommandIntent
    VOICE_AVAILABLE = True
    print("✅ Voice recognition modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import voice recognition modules: {e}")
    VOICE_AVAILABLE = False

def test_microphone_basic():
    """Test basic microphone functionality."""
    print("\n🔊 Testing Microphone...")
    print("-" * 30)
    
    if not VOICE_AVAILABLE:
        print("❌ Voice components not available")
        return False
    
    try:
        # List available audio devices
        devices = list_audio_devices()
        print(f"📱 Found {len(devices)} audio input devices:")
        for device in devices:
            print(f"   - {device['name']} ({device['channels']} channels, {device['sample_rate']} Hz)")
        
        # Test microphone
        print("\n🎤 Testing microphone recording...")
        result = test_microphone()
        
        if result:
            print("✅ Microphone test PASSED - audio detected")
            return True
        else:
            print("❌ Microphone test FAILED - no audio detected")
            print("   Check microphone permissions and hardware")
            return False
            
    except Exception as e:
        print(f"❌ Microphone test error: {e}")
        return False

def test_command_parsing():
    """Test command parsing functionality."""
    print("\n🤖 Testing Command Parsing...")
    print("-" * 30)
    
    if not VOICE_AVAILABLE:
        print("❌ Voice components not available")
        return False
    
    try:
        # Create command parser
        parser = create_parser(use_nlp=False)
        print("✅ Command parser created successfully")
        
        # Test commands
        test_commands = [
            "describe data",
            "filter age greater than 25",
            "sort by name ascending",
            "visualize column sales",
            "sum of revenue",
            "show data",
            "create bar chart for age",
            "this is not a valid command"
        ]
        
        print(f"\n🧪 Testing {len(test_commands)} commands:")
        
        success_count = 0
        for i, cmd in enumerate(test_commands, 1):
            result = parser.parse(cmd)
            
            if result.intent != CommandIntent.UNKNOWN:
                success_count += 1
                status = "✅ PARSED"
                details = []
                if result.column:
                    details.append(f"column={result.column}")
                if result.operation:
                    details.append(f"op={result.operation.value}")
                if result.value:
                    details.append(f"value={result.value}")
                if result.chart_type:
                    details.append(f"chart={result.chart_type.value}")
                
                detail_str = f" ({', '.join(details)})" if details else ""
                print(f"   {i}. '{cmd}' → {result.intent.value}{detail_str}")
            else:
                status = "❌ UNKNOWN"
                print(f"   {i}. '{cmd}' → {status}")
        
        success_rate = (success_count / len(test_commands)) * 100
        print(f"\n📊 Parsing Results: {success_count}/{len(test_commands)} successful ({success_rate:.1f}%)")
        
        return success_count > 0
        
    except Exception as e:
        print(f"❌ Command parsing test error: {e}")
        return False

def test_voice_handler():
    """Test voice handler initialization."""
    print("\n🎤 Testing Voice Handler...")
    print("-" * 30)
    
    if not VOICE_AVAILABLE:
        print("❌ Voice components not available")
        return False
    
    try:
        # Create voice handler
        voice_handler = VoiceHandler()
        print("✅ Voice handler created")
        
        # Configure for testing
        config = {
            "audio": {
                "silence_duration": 2.0,
                "max_recording_duration": 5.0,
                "silence_threshold": 0.02,
                "min_recording_duration": 0.5
            },
            "whisper": {
                "model_name": "base",
                "temperature": 0.0
            }
        }
        
        print("🔧 Initializing voice handler...")
        if voice_handler.initialize(config):
            print("✅ Voice handler initialized successfully")
            print(f"   - Whisper model: {config['whisper']['model_name']}")
            print(f"   - Silence threshold: {config['audio']['silence_threshold']}")
            print(f"   - Max duration: {config['audio']['max_recording_duration']}s")
            
            # Cleanup
            voice_handler.cleanup()
            return True
        else:
            print("❌ Voice handler initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Voice handler test error: {e}")
        return False

def test_full_pipeline():
    """Test the full voice recognition pipeline interactively."""
    print("\n🚀 Testing Full Pipeline...")
    print("-" * 30)
    
    if not VOICE_AVAILABLE:
        print("❌ Voice components not available")
        return False
    
    try:
        # Create components
        voice_handler = VoiceHandler()
        command_parser = create_parser(use_nlp=False)
        
        # Configure voice handler
        config = {
            "audio": {
                "silence_duration": 1.5,
                "max_recording_duration": 8.0,
                "silence_threshold": 0.02,
                "min_recording_duration": 0.5
            },
            "whisper": {
                "model_name": "base",
                "temperature": 0.0
            }
        }
        
        if not voice_handler.initialize(config):
            print("❌ Failed to initialize voice handler")
            return False
        
        print("✅ Full pipeline ready!")
        print("\n📋 Interactive Test Instructions:")
        print("   1. Press ENTER to start recording")
        print("   2. Speak clearly: 'describe data' or 'filter age greater than 25'")
        print("   3. Wait for processing")
        print("   4. Check transcription and parsing results")
        print("   5. Type 'quit' to exit")
        
        # Track results
        transcription_received = False
        parsed_result = None
        
        def on_transcription(text, confidence):
            nonlocal transcription_received, parsed_result
            transcription_received = True
            
            print(f"\n🎤 TRANSCRIPTION: '{text}' (confidence: {confidence:.2f})")
            
            # Parse the command
            parsed_result = command_parser.parse(text)
            print(f"🤖 PARSED: {parsed_result.intent.value}")
            
            if parsed_result.intent != CommandIntent.UNKNOWN:
                details = []
                if parsed_result.column:
                    details.append(f"column={parsed_result.column}")
                if parsed_result.operation:
                    details.append(f"operation={parsed_result.operation.value}")
                if parsed_result.value:
                    details.append(f"value={parsed_result.value}")
                
                if details:
                    print(f"   └─ {', '.join(details)}")
                
                print("✅ COMMAND RECOGNIZED!")
            else:
                print("⚠️ Command not recognized")
        
        def on_error(error):
            print(f"❌ ERROR: {error}")
        
        # Connect callbacks
        voice_handler.transcription_received.connect(on_transcription)
        voice_handler.error_occurred.connect(on_error)
        
        # Interactive loop
        while True:
            user_input = input("\nPress ENTER to record (or 'quit' to exit): ").strip().lower()
            
            if user_input == 'quit':
                break
            
            print("🎤 Starting recording... speak now!")
            transcription_received = False
            
            if voice_handler.start_listening():
                # Wait for transcription
                start_time = time.time()
                while not transcription_received and (time.time() - start_time) < 15:
                    time.sleep(0.1)
                
                if not transcription_received:
                    print("⏱️ Timeout - no transcription received")
                    voice_handler.stop_listening()
            else:
                print("❌ Failed to start recording")
        
        # Cleanup
        voice_handler.cleanup()
        print("\n🏁 Pipeline test completed")
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test error: {e}")
        return False

def main():
    """Run all voice recognition tests."""
    print("🧪 Voice Recognition Pipeline Test")
    print("=" * 50)
    
    if not VOICE_AVAILABLE:
        print("❌ Voice recognition not available - check dependencies:")
        print("   pip install sounddevice scipy openai-whisper")
        print("   brew install ffmpeg")
        return
    
    # Run tests
    tests = [
        ("Microphone", test_microphone_basic),
        ("Command Parsing", test_command_parsing),
        ("Voice Handler", test_voice_handler),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print(f"{'='*50}")
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nOverall: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Voice recognition should work.")
        
        # Offer interactive test
        interactive = input("\nWould you like to run the interactive pipeline test? (y/n): ").strip().lower()
        if interactive == 'y':
            test_full_pipeline()
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
        print("Common issues:")
        print("- Microphone permissions not granted")
        print("- ffmpeg not installed")
        print("- Audio devices not available")

if __name__ == "__main__":
    main() 