#!/usr/bin/env python3
"""
UI Integration Test
==================

Simple test to verify the enhanced classifier adapter works correctly
with the UI before launching the full application.
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_adapter():
    """Test the gesture classifier adapter."""
    print("🧪 Testing Enhanced Classifier Adapter Integration...")
    print("=" * 50)
    
    try:
        # Import the adapter (same way UI does)
        from core.gesture_classifier_adapter import GestureClassifier
        
        print("✅ Import successful!")
        
        # Initialize classifier (same way UI does)
        print("🔄 Initializing classifier...")
        classifier = GestureClassifier(
            model_type='knn', 
            use_feature_selection=True, 
            n_features=50
        )
        
        print("✅ Initialization successful!")
        
        # Test training (same way UI does)
        print("🔄 Testing training...")
        classifier.train()
        
        print("✅ Training successful!")
        
        # Test prediction with sample landmarks
        print("🔄 Testing prediction...")
        
        # Create sample landmarks (21 hand landmarks)
        sample_landmarks = []
        for i in range(21):
            landmark = {
                'id': i,
                'x': 100 + i * 10,
                'y': 200 + i * 5,
                'z': 0.0,
                'visibility': 1.0
            }
            sample_landmarks.append(landmark)
        
        # Test prediction
        result = classifier.predict_gesture(sample_landmarks)
        
        print(f"✅ Prediction successful!")
        print(f"   Gesture: {result['gesture']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Type: {result['type']}")
        
        # Test model info
        info = classifier.get_gesture_info()
        print(f"✅ Model info: {info.get('model_type', 'unknown')}")
        
        print("\n🎉 All tests passed!")
        print("✨ Enhanced classifier is ready for UI integration!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you've trained an enhanced model:")
        print("   python run_enhanced_training.py")
        print("2. Check if all dependencies are installed")
        print("3. Verify file paths are correct")
        return False

def test_ui_compatibility():
    """Test UI compatibility without launching full UI."""
    print("\n🖥️  Testing UI Compatibility...")
    print("=" * 30)
    
    try:
        # Test the import path that UI uses
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
        
        # This mimics what the UI does
        from core.hand_tracking.core.gesture_classifier_adapter import GestureClassifier
        
        print("✅ UI import path works!")
        
        # Quick functionality test
        classifier = GestureClassifier()
        print("✅ UI-style initialization works!")
        
        return True
        
    except Exception as e:
        print(f"❌ UI compatibility test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Enhanced Classifier UI Integration Test")
    print("=" * 60)
    
    success = test_adapter()
    
    if success:
        ui_success = test_ui_compatibility()
        
        if ui_success:
            print(f"\n🎉 INTEGRATION SUCCESSFUL!")
            print("✨ Your UI will now use the enhanced classifier automatically!")
            print("\n🚀 Next steps:")
            print("1. Launch your UI normally")
            print("2. Enable gesture classification in the camera panel")
            print("3. Enjoy much better gesture recognition accuracy!")
        else:
            print(f"\n⚠️  Adapter works but UI compatibility needs fixing")
    else:
        print(f"\n❌ Integration test failed - please check the setup") 