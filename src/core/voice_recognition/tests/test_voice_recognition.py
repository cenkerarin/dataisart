"""
Comprehensive Voice Recognition Test Suite
=========================================

Complete testing suite for voice recognition functionality including:
- Audio device detection
- Microphone testing
- Audio recording and saving
- Whisper transcription
- Voice handler integration
- Debugging utilities

Usage:
    # From project root:
    python src/core/voice_recognition/tests/test_voice_recognition.py

    # From voice_recognition directory:
    python tests/test_voice_recognition.py
"""

import sys
import os
import time
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Fix import paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_RECOGNITION_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(VOICE_RECOGNITION_DIR)

try:
    import sounddevice as sd
    from scipy.io import wavfile
    import whisper
    from voice_handler import VoiceHandler, test_microphone, list_audio_devices
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Run: pip install sounddevice scipy openai-whisper")
    DEPENDENCIES_AVAILABLE = False
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoiceRecognitionTestSuite:
    """Comprehensive test suite for voice recognition system."""
    
    def __init__(self):
        self.test_results = {
            'audio_devices': False,
            'microphone_basic': False,
            'microphone_levels': False,
            'audio_recording': False,
            'whisper_transcription': False,
            'voice_handler_init': False,
            'voice_handler_integration': False
        }
        self.temp_files = []
        self.transcriptions = []
        self.errors = []
    
    def print_header(self, title: str):
        """Print a formatted test section header."""
        print("\n" + "=" * 60)
        print(f"🧪 {title}")
        print("=" * 60)
    
    def print_result(self, test_name: str, passed: bool, details: str = ""):
        """Print test result with formatting."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if details:
            print(f"   {details}")
        self.test_results[test_name] = passed
    
    def test_audio_devices(self) -> bool:
        """Test 1: Audio device detection and listing."""
        self.print_header("AUDIO DEVICE DETECTION TEST")
        
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            
            print(f"Found {len(devices)} total audio devices")
            print(f"Found {len(input_devices)} input devices:")
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    print(f"  • [{i}] {device['name']} - {device['max_input_channels']}ch @ {device['default_samplerate']:.0f}Hz")
            
            # Test our utility function
            util_devices = list_audio_devices()
            
            success = len(input_devices) > 0 and len(util_devices) > 0
            details = f"Found {len(input_devices)} input devices" if success else "No input devices detected"
            self.print_result('audio_devices', success, details)
            
            return success
            
        except Exception as e:
            self.print_result('audio_devices', False, f"Error: {e}")
            return False
    
    def test_microphone_basic(self) -> bool:
        """Test 2: Basic microphone functionality."""
        self.print_header("BASIC MICROPHONE TEST")
        
        try:
            print("Testing microphone for 2 seconds...")
            print("Please make some noise (speak, clap, etc.)")
            
            success = test_microphone()
            details = "Microphone detected audio" if success else "No audio detected - check microphone"
            self.print_result('microphone_basic', success, details)
            
            return success
            
        except Exception as e:
            self.print_result('microphone_basic', False, f"Error: {e}")
            return False
    
    def test_microphone_levels(self) -> bool:
        """Test 3: Microphone level monitoring."""
        self.print_header("MICROPHONE LEVEL TEST")
        
        try:
            duration = 3
            sample_rate = 16000
            
            print(f"Recording for {duration} seconds...")
            print("Speak into your microphone - watch the levels!")
            
            levels = []
            max_level = 0
            
            def audio_callback(indata, frames, time, status):
                nonlocal max_level
                if status:
                    print(f"Audio status: {status}")
                
                rms = np.sqrt(np.mean(indata**2))
                levels.append(rms)
                max_level = max(max_level, rms)
                
                # Visual level indicator
                level_bars = int(rms * 50)
                bar = "█" * level_bars + "░" * (50 - level_bars)
                print(f"\rLevel: |{bar}| {rms:.4f}", end="", flush=True)
            
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                callback=audio_callback,
                dtype='float32'
            ):
                time.sleep(duration)
            
            print(f"\nMax level recorded: {max_level:.4f}")
            
            success = max_level > 0.001  # Basic threshold for audio detection
            details = f"Max RMS: {max_level:.4f}" + (" - Good levels!" if success else " - Levels too low")
            self.print_result('microphone_levels', success, details)
            
            return success
            
        except Exception as e:
            self.print_result('microphone_levels', False, f"Error: {e}")
            return False
    
    def test_audio_recording(self) -> bool:
        """Test 4: Audio recording and file saving."""
        self.print_header("AUDIO RECORDING TEST")
        
        try:
            duration = 3
            sample_rate = 16000
            filename = "test_recording.wav"
            
            print(f"Recording {duration} seconds to {filename}...")
            print("Speak clearly into your microphone!")
            
            # Record audio
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            
            # Analyze recording
            rms = np.sqrt(np.mean(recording**2))
            max_amplitude = np.max(np.abs(recording))
            
            print(f"Recording quality:")
            print(f"  RMS Level: {rms:.4f}")
            print(f"  Max Amplitude: {max_amplitude:.4f}")
            print(f"  Duration: {len(recording) / sample_rate:.2f}s")
            
            # Save to file
            recording_int16 = (recording * 32767).astype(np.int16)
            wavfile.write(filename, sample_rate, recording_int16)
            self.temp_files.append(filename)
            
            # Verify file was created
            file_exists = Path(filename).exists()
            
            success = rms > 0.001 and file_exists
            details = f"RMS: {rms:.4f}, File saved: {file_exists}"
            self.print_result('audio_recording', success, details)
            
            return success
            
        except Exception as e:
            self.print_result('audio_recording', False, f"Error: {e}")
            return False
    
    def test_whisper_transcription(self) -> bool:
        """Test 5: Whisper transcription functionality."""
        self.print_header("WHISPER TRANSCRIPTION TEST")
        
        if not any(Path(f).exists() for f in self.temp_files):
            self.print_result('whisper_transcription', False, "No audio file available")
            return False
        
        try:
            audio_file = next(f for f in self.temp_files if Path(f).exists())
            
            print("Loading Whisper model...")
            model = whisper.load_model("base")
            print("✅ Whisper model loaded successfully!")
            
            print(f"Transcribing {audio_file}...")
            result = model.transcribe(audio_file)
            
            transcribed_text = result["text"].strip()
            language = result.get("language", "unknown")
            
            print("\n" + "=" * 30)
            print("TRANSCRIPTION RESULT:")
            print("=" * 30)
            print(f"Text: '{transcribed_text}'")
            print(f"Language: {language}")
            print("=" * 30)
            
            success = len(transcribed_text) > 0
            details = f"Text length: {len(transcribed_text)} chars" if success else "No text transcribed"
            self.print_result('whisper_transcription', success, details)
            
            return success
            
        except Exception as e:
            self.print_result('whisper_transcription', False, f"Error: {e}")
            return False
    
    def test_voice_handler_init(self) -> bool:
        """Test 6: Voice handler initialization."""
        self.print_header("VOICE HANDLER INITIALIZATION TEST")
        
        try:
            voice_handler = VoiceHandler()
            
            config = {
                "audio": {
                    "sample_rate": 16000,
                    "dtype": "float32",
                    "silence_threshold": 0.01,
                    "silence_duration": 1.5,
                    "max_recording_duration": 5.0,
                    "min_recording_duration": 0.5
                },
                "whisper": {
                    "model_name": "base",
                    "device": "cpu",
                    "language": None,
                    "temperature": 0.0
                }
            }
            
            success = voice_handler.initialize(config)
            details = "Initialized with custom config" if success else "Initialization failed"
            self.print_result('voice_handler_init', success, details)
            
            # Cleanup
            if hasattr(voice_handler, 'cleanup'):
                voice_handler.cleanup()
            
            return success
            
        except Exception as e:
            self.print_result('voice_handler_init', False, f"Error: {e}")
            return False
    
    def test_voice_handler_integration(self) -> bool:
        """Test 7: Voice handler integration (quick test)."""
        self.print_header("VOICE HANDLER INTEGRATION TEST")
        
        try:
            voice_handler = VoiceHandler()
            
            # Set up transcription callback
            def on_transcription(text: str, confidence: float):
                self.transcriptions.append((text, confidence, time.time()))
                print(f"🎯 Transcription: '{text}' (Confidence: {confidence:.1%})")
            
            def on_error(error_msg: str):
                self.errors.append(error_msg)
                print(f"❌ Error: {error_msg}")
            
            # Initialize with shorter timeouts for testing
            config = {
                "audio": {
                    "max_recording_duration": 3.0,
                    "silence_duration": 1.0,
                    "silence_threshold": 0.01
                },
                "whisper": {"model_name": "base"}
            }
            
            init_success = voice_handler.initialize(config)
            if not init_success:
                self.print_result('voice_handler_integration', False, "Failed to initialize")
                return False
            
            # Connect signals/callbacks
            try:
                voice_handler.transcription_received.connect(on_transcription)
                voice_handler.error_occurred.connect(on_error)
                print("✅ Connected PyQt signals")
            except Exception:
                print("⚠️  PyQt signals not available, using callbacks")
            
            print("\n🎤 Quick integration test:")
            print("Speak for 2-3 seconds, then be quiet...")
            print("(This is a quick test - may timeout if no speech)")
            
            start_success = voice_handler.start_listening(callback=on_transcription)
            if not start_success:
                self.print_result('voice_handler_integration', False, "Failed to start listening")
                return False
            
            # Wait for a short time
            time.sleep(5)
            
            # Stop and cleanup
            voice_handler.stop_listening()
            voice_handler.cleanup()
            
            # Evaluate results
            has_transcriptions = len(self.transcriptions) > 0
            has_errors = len(self.errors) > 0
            
            if has_transcriptions:
                details = f"Got {len(self.transcriptions)} transcription(s)"
                success = True
            elif has_errors:
                details = f"Errors occurred: {len(self.errors)}"
                success = False
            else:
                details = "No transcriptions (may need louder speech or longer recording)"
                success = True  # Not necessarily a failure if no speech was detected
            
            self.print_result('voice_handler_integration', success, details)
            return success
            
        except Exception as e:
            self.print_result('voice_handler_integration', False, f"Error: {e}")
            return False
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during testing."""
        for filepath in self.temp_files:
            try:
                if Path(filepath).exists():
                    Path(filepath).unlink()
                    print(f"🧹 Cleaned up: {filepath}")
            except Exception as e:
                print(f"⚠️  Could not remove {filepath}: {e}")
    
    def print_final_summary(self):
        """Print final test summary."""
        print("\n" + "=" * 60)
        print("📋 FINAL TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        print(f"Tests run: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, passed in self.test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status}: {test_name}")
        
        if self.transcriptions:
            print(f"\n🎯 Transcriptions received ({len(self.transcriptions)}):")
            for text, confidence, timestamp in self.transcriptions:
                print(f"  • '{text}' (Confidence: {confidence:.1%})")
        
        if self.errors:
            print(f"\n❌ Errors encountered ({len(self.errors)}):")
            for error in self.errors[-3:]:  # Show last 3 errors
                print(f"  • {error}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if passed_tests == total_tests:
            print("  🎉 All tests passed! Voice recognition system is working properly.")
        elif self.test_results.get('audio_devices') and self.test_results.get('microphone_basic'):
            print("  🔧 Basic audio is working. Issues may be with:")
            print("    - Whisper model loading")
            print("    - PyQt threading")
            print("    - Audio levels (speak louder)")
        else:
            print("  🔴 Basic audio issues detected:")
            print("    - Check microphone permissions")
            print("    - Verify microphone is connected")
            print("    - Test with system audio settings")
        
        return passed_tests == total_tests
    
    def run_all_tests(self) -> bool:
        """Run all tests in sequence."""
        print("🚀 Starting Comprehensive Voice Recognition Test Suite")
        
        if not DEPENDENCIES_AVAILABLE:
            print("❌ Cannot run tests - missing dependencies")
            return False
        
        try:
            # Run all tests
            self.test_audio_devices()
            self.test_microphone_basic()
            self.test_microphone_levels()
            self.test_audio_recording()
            self.test_whisper_transcription()
            self.test_voice_handler_init()
            self.test_voice_handler_integration()
            
        except KeyboardInterrupt:
            print("\n\n🛑 Tests interrupted by user")
        except Exception as e:
            print(f"\n\n💥 Unexpected error during testing: {e}")
        finally:
            # Always cleanup
            self.cleanup_temp_files()
            return self.print_final_summary()

def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Recognition Test Suite")
    parser.add_argument('--quick', action='store_true', help='Run only basic tests')
    parser.add_argument('--test', choices=['devices', 'mic', 'record', 'whisper', 'handler'], 
                       help='Run specific test only')
    
    args = parser.parse_args()
    
    test_suite = VoiceRecognitionTestSuite()
    
    if args.test:
        # Run specific test
        test_methods = {
            'devices': test_suite.test_audio_devices,
            'mic': test_suite.test_microphone_basic,
            'record': test_suite.test_audio_recording,
            'whisper': test_suite.test_whisper_transcription,
            'handler': test_suite.test_voice_handler_integration
        }
        
        if args.test in test_methods:
            success = test_methods[args.test]()
            test_suite.cleanup_temp_files()
            sys.exit(0 if success else 1)
    
    elif args.quick:
        # Quick test mode
        success = (test_suite.test_audio_devices() and 
                  test_suite.test_microphone_basic() and
                  test_suite.test_voice_handler_init())
        test_suite.cleanup_temp_files()
        print(f"\n🏃 Quick test {'PASSED' if success else 'FAILED'}")
        sys.exit(0 if success else 1)
    
    else:
        # Full test suite
        success = test_suite.run_all_tests()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 