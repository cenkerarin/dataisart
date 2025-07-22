#!/usr/bin/env python3
"""
Quick Enhanced Classifier Test
==============================

Quick test to verify the enhanced classifier works with realistic data.
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_enhanced_classifier():
    """Test the enhanced classifier with realistic data."""
    print("🚀 Quick Enhanced Classifier Test")
    print("=" * 40)
    
    try:
        from core.gesture_classifier_adapter import GestureClassifier
        
        # Initialize
        print("🔄 Initializing...")
        classifier = GestureClassifier()
        
        # Train
        print("🔄 Training...")
        classifier.train()
        
        # Test with realistic landmarks (no NaN issues)
        print("🔄 Testing with realistic data...")
        
        realistic_landmarks = []
        for i in range(21):
            landmark = {
                'id': i,
                'x': 300.0 + i * 15.0,    # Realistic screen coordinates
                'y': 250.0 + i * 8.0,     # Realistic screen coordinates  
                'z': -0.1 + i * 0.01,     # Realistic depth values
                'visibility': 0.95        # High visibility
            }
            realistic_landmarks.append(landmark)
        
        result = classifier.predict_gesture(realistic_landmarks)
        
        print(f"✅ Prediction result:")
        print(f"   Gesture: {result['gesture']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Type: {result['type']}")
        
        if result['gesture'] != 'error':
            print("\n🎉 SUCCESS! Enhanced classifier is working!")
            print("✨ Ready for UI integration!")
            return True
        else:
            print("\n⚠️  Still getting error - but no crash")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_enhanced_classifier() 