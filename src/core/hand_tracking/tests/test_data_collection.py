"""
Test Data Collection and Training Pipeline
=========================================

Simple test to verify the data collection and training works correctly.
"""

import os
from tools.data_collector import GestureDataCollector
from core.gesture_classifier import GestureClassifier

def test_data_collection():
    """Test the data collection workflow."""
    print("Testing Data Collection & Training Pipeline")
    print("=" * 50)
    
    # Initialize collector
    collector = GestureDataCollector()
    
    if not collector.initialize():
        print("❌ Failed to initialize hand detector")
        return False
    
    print("✅ Hand detector initialized successfully")
    
    # Check if training data exists
    data_file = "data/gesture_data/training_data.json"
    
    if os.path.exists(data_file):
        print(f"✅ Training data found: {data_file}")
        
        # Test loading data
        try:
            collector.load_data(data_file)
            print("✅ Successfully loaded training data")
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            return False
    else:
        print("📋 No training data found. You can collect it by running:")
        print("   python data_collector.py")
    
    # Test classifier
    print("\n" + "=" * 30)
    print("Testing Gesture Classifier")
    print("=" * 30)
    
    classifier = GestureClassifier()
    
    try:
        # This will use real data if available, synthetic otherwise
        classifier.train()
        print("✅ Classifier trained successfully")
        
        # Test prediction with synthetic data
        test_landmarks = classifier._generate_synthetic_landmarks('pointing')
        result = classifier.predict_gesture(test_landmarks)
        
        print(f"✅ Test prediction: {result['gesture']} (confidence: {result['confidence']:.2f})")
        
    except Exception as e:
        print(f"❌ Error training classifier: {e}")
        return False
    
    return True

def main():
    """Main test function."""
    print("🤖 Hand Tracking Data Collection Test")
    print("=" * 50)
    
    success = test_data_collection()
    
    if success:
        print("\n✅ All tests passed!")
        print("\nNext steps:")
        print("1. Run 'python data_collector.py' to collect real gesture data")
        print("2. Run 'python gesture_classification_demo.py' to test real-time recognition")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 