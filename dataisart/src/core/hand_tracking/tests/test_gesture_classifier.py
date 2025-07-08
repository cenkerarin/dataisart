"""
Test Gesture Classifier
=======================

Test script to demonstrate gesture classification functionality,
model training, saving/loading, and adding new gestures.
"""

import numpy as np
import logging
from ..core.gesture_classifier import GestureClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test basic gesture classifier functionality."""
    print("="*60)
    print("TESTING BASIC FUNCTIONALITY")
    print("="*60)
    
    # Create classifier
    classifier = GestureClassifier(model_type='knn', temporal_window=5)
    
    # Show initial state
    info = classifier.get_gesture_info()
    print(f"Model type: {info['model_type']}")
    print(f"Supported gestures: {info['supported_gestures']}")
    print(f"Is trained: {info['is_trained']}")
    
    # Train the model
    print("\nTraining classifier...")
    classifier.train()
    
    # Test with synthetic data
    print("\nTesting gesture prediction...")
    test_gestures = ['pointing', 'fist', 'open_hand', 'pinch']
    
    for gesture_name in test_gestures:
        # Generate test landmarks
        landmarks = classifier._generate_synthetic_landmarks(gesture_name)
        
        # Predict
        result = classifier.predict_gesture(landmarks)
        
        print(f"\nTest gesture: {gesture_name}")
        print(f"Predicted: {result['gesture']} (confidence: {result['confidence']:.3f})")
        
        if result.get('probabilities'):
            print("Top 3 probabilities:")
            sorted_probs = sorted(result['probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            for pred_gesture, prob in sorted_probs:
                print(f"  {pred_gesture}: {prob:.3f}")

def test_feature_extraction():
    """Test feature extraction functionality."""
    print("\n" + "="*60)
    print("TESTING FEATURE EXTRACTION")
    print("="*60)
    
    classifier = GestureClassifier()
    
    # Generate sample landmarks
    sample_landmarks = []
    for i in range(21):
        landmark = {
            'id': i,
            'x': 100 + i * 10,
            'y': 200 + i * 5,
            'z': 0.1 * i,
            'visibility': 1.0
        }
        sample_landmarks.append(landmark)
    
    # Extract features
    features = classifier.extract_features(sample_landmarks)
    
    print(f"Number of landmarks: {len(sample_landmarks)}")
    print(f"Feature vector size: {len(features)}")
    print(f"Feature vector (first 10): {features[:10]}")
    
    # Test with different gestures
    gestures_to_test = ['neutral', 'pointing', 'fist']
    print(f"\nFeature comparison for different gestures:")
    
    for gesture in gestures_to_test:
        landmarks = classifier._generate_synthetic_landmarks(gesture)
        features = classifier.extract_features(landmarks)
        print(f"{gesture:10}: mean={np.mean(features):.3f}, std={np.std(features):.3f}")

def test_model_persistence():
    """Test saving and loading models."""
    print("\n" + "="*60)
    print("TESTING MODEL PERSISTENCE")
    print("="*60)
    
    # Train a model
    classifier1 = GestureClassifier(model_type='rf')
    classifier1.train()
    
    # Test prediction before saving
    test_landmarks = classifier1._generate_synthetic_landmarks('pointing')
    result_before = classifier1.predict_gesture(test_landmarks)
    print(f"Prediction before save: {result_before['gesture']} ({result_before['confidence']:.3f})")
    
    # Save model
    model_path = "test_gesture_model.pkl"
    classifier1.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Load model in new classifier
    classifier2 = GestureClassifier()
    classifier2.load_model(model_path)
    
    # Test prediction after loading
    result_after = classifier2.predict_gesture(test_landmarks)
    print(f"Prediction after load: {result_after['gesture']} ({result_after['confidence']:.3f})")
    
    # Verify they're the same
    if (result_before['gesture'] == result_after['gesture'] and 
        abs(result_before['confidence'] - result_after['confidence']) < 0.001):
        print("✓ Model persistence test PASSED")
    else:
        print("✗ Model persistence test FAILED")
    
    # Cleanup
    import os
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Cleaned up {model_path}")

def test_adding_new_gestures():
    """Test adding new gesture classes."""
    print("\n" + "="*60)
    print("TESTING ADDING NEW GESTURES")
    print("="*60)
    
    classifier = GestureClassifier()
    
    # Show initial gestures
    info = classifier.get_gesture_info()
    print(f"Initial gestures: {info['supported_gestures']}")
    
    # Add new gesture
    classifier.add_gesture("wave", 10)
    classifier.add_gesture("stop_sign", 11)
    
    # Show updated gestures
    info = classifier.get_gesture_info()
    print(f"Updated gestures: {info['supported_gestures']}")
    print(f"Total gestures: {info['total_gestures']}")
    
    # Note: Model would need retraining with new data
    print("Note: Model would need retraining with new gesture data")

def test_swipe_detection():
    """Test temporal swipe detection."""
    print("\n" + "="*60)
    print("TESTING SWIPE DETECTION")
    print("="*60)
    
    classifier = GestureClassifier(temporal_window=8)
    classifier.train()
    
    # Simulate hand movement for swipe right
    print("Simulating swipe right gesture...")
    
    for i in range(10):
        # Create landmarks moving from left to right
        landmarks = []
        base_x = 100 + i * 20  # Moving right
        base_y = 300
        
        for j in range(21):
            landmark = {
                'id': j,
                'x': base_x + j * 5,
                'y': base_y + j * 2,
                'z': 0.0,
                'visibility': 1.0
            }
            landmarks.append(landmark)
        
        result = classifier.predict_gesture(landmarks)
        
        if result.get('type') == 'temporal':
            print(f"Frame {i}: {result['gesture']} (confidence: {result['confidence']:.3f})")
            break
        elif i < 5:
            print(f"Frame {i}: {result['gesture']} (building temporal history...)")
    
    # Simulate swipe left
    print("\nSimulating swipe left gesture...")
    classifier.hand_center_history.clear()  # Reset history
    
    for i in range(10):
        # Create landmarks moving from right to left
        landmarks = []
        base_x = 300 - i * 20  # Moving left
        base_y = 300
        
        for j in range(21):
            landmark = {
                'id': j,
                'x': base_x + j * 5,
                'y': base_y + j * 2,
                'z': 0.0,
                'visibility': 1.0
            }
            landmarks.append(landmark)
        
        result = classifier.predict_gesture(landmarks)
        
        if result.get('type') == 'temporal':
            print(f"Frame {i}: {result['gesture']} (confidence: {result['confidence']:.3f})")
            break
        elif i < 5:
            print(f"Frame {i}: {result['gesture']} (building temporal history...)")

def test_model_comparison():
    """Compare different ML models."""
    print("\n" + "="*60)
    print("TESTING MODEL COMPARISON")
    print("="*60)
    
    models = ['knn', 'rf']
    results = {}
    
    for model_type in models:
        print(f"\nTesting {model_type.upper()} model...")
        
        classifier = GestureClassifier(model_type=model_type)
        classifier.train()
        
        # Test accuracy on a few samples
        correct = 0
        total = 20
        
        test_gestures = ['pointing', 'fist', 'open_hand', 'pinch']
        
        for _ in range(total):
            # Pick random gesture
            true_gesture = np.random.choice(test_gestures)
            landmarks = classifier._generate_synthetic_landmarks(true_gesture)
            prediction = classifier.predict_gesture(landmarks)
            
            if prediction['gesture'] == true_gesture:
                correct += 1
        
        accuracy = correct / total
        results[model_type] = accuracy
        print(f"{model_type.upper()} accuracy: {accuracy:.3f}")
    
    print(f"\nModel comparison:")
    for model, acc in results.items():
        print(f"  {model.upper()}: {acc:.3f}")

def run_all_tests():
    """Run all test functions."""
    print("GESTURE CLASSIFIER TEST SUITE")
    print("="*60)
    
    try:
        test_basic_functionality()
        test_feature_extraction()
        test_model_persistence()
        test_adding_new_gestures()
        test_swipe_detection()
        test_model_comparison()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests() 