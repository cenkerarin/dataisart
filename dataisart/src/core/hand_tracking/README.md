# Hand Tracking Module

A comprehensive gesture recognition and hand tracking system using MediaPipe and machine learning.

## 🏗️ Project Structure

```
src/core/hand_tracking/
├── core/                           # Core functionality
│   ├── __init__.py
│   ├── gesture_detector.py         # Hand detection using MediaPipe
│   └── gesture_classifier.py       # ML-based gesture classification
├── data/                           # Data collection and management
│   ├── __init__.py
│   ├── data_collector.py          # Tool for collecting real gesture data
│   └── gestures/                   # Directory for collected gesture data
├── training/                       # Model training utilities
│   ├── __init__.py
│   └── train_improved_model.py     # Advanced training pipeline
├── demos/                          # Example applications
│   ├── __init__.py
│   ├── gesture_classification_demo.py  # Real-time gesture recognition demo
│   ├── demo_hand_tracking.py       # Basic hand tracking demo
│   └── usage_example.py           # Simple usage examples
├── tests/                          # Test files
│   ├── __init__.py
│   └── test_gesture_classifier.py  # Unit tests
├── README.md                       # This file
└── example_hand_data_structure.json  # Sample data format
```

## 🚀 Quick Start

### 1. Import the Main Classes

```python
from src.core.hand_tracking import GestureDetector, GestureClassifier, GestureDataCollector
```

### 2. Basic Hand Detection

```python
from src.core.hand_tracking.core import GestureDetector
import cv2

# Initialize detector
detector = GestureDetector()
config = {
    "max_num_hands": 2,
    "min_detection_confidence": 0.7
}
detector.initialize(config)

# Process frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
results = detector.detect_hands(frame)

if results["hands_detected"]:
    print(f"Detected {len(results['hands'])} hands")
    for hand in results["hands"]:
        print(f"Hand: {hand['handedness']['label']}")
```

### 3. Gesture Classification

```python
from src.core.hand_tracking.core import GestureClassifier

# Initialize and train classifier
classifier = GestureClassifier()
classifier.train()  # Uses real data if available, synthetic otherwise

# Classify gesture from landmarks
result = classifier.predict_gesture(landmarks)
print(f"Gesture: {result['gesture']} (confidence: {result['confidence']:.2f})")
```

## 📊 Data Collection & Training

### Collect Real Training Data

```bash
cd src/core/hand_tracking/data
python data_collector.py
```

This interactive tool helps you collect real gesture samples for training.

### Train Improved Model

```bash
cd src/core/hand_tracking/training  
python train_improved_model.py
```

This advanced training pipeline:
- Compares multiple ML algorithms
- Performs hyperparameter tuning
- Provides detailed evaluation metrics
- Saves optimized models

## 🎮 Demo Applications

### Real-time Gesture Recognition

```bash
cd src/core/hand_tracking/demos
python gesture_classification_demo.py
```

Features:
- Live gesture recognition
- Multiple gesture types (static + temporal)
- Real-time statistics
- Interactive controls

### Basic Hand Tracking

```bash
cd src/core/hand_tracking/demos  
python demo_hand_tracking.py
```

Simple demonstration of hand detection and landmark extraction.

## 🧪 Testing

```bash
cd src/core/hand_tracking/tests
python test_gesture_classifier.py
```

Comprehensive test suite covering:
- Feature extraction
- Model training/loading
- Gesture prediction
- Data persistence

## 🎯 Supported Gestures

### Static Gestures
- **Neutral**: Relaxed hand
- **Pointing**: Index finger extended
- **Fist**: Closed hand
- **Open Hand**: All fingers extended
- **Peace Sign**: Index and middle fingers extended
- **Thumbs Up**: Thumb extended, others closed
- **Pinch**: Thumb and index finger close

### Temporal Gestures
- **Swipe Left**: Horizontal hand movement left
- **Swipe Right**: Horizontal hand movement right

## 🔧 Architecture

### Core Components

#### GestureDetector
- Uses MediaPipe for robust hand tracking
- Extracts 21 hand landmarks per hand
- Provides handedness classification
- Calculates bounding boxes

#### GestureClassifier  
- Advanced feature engineering (80+ features)
- Support for multiple ML algorithms
- Real-time temporal gesture detection
- Model persistence and loading

#### GestureDataCollector
- Interactive data collection interface
- Supports multiple users and variations
- Structured data storage format
- Quality control and validation

### Feature Engineering

The classifier extracts comprehensive features:

1. **Normalized Coordinates**: Scale/position invariant
2. **Distance Features**: Between key landmarks
3. **Angle Features**: Joint angles and finger spread
4. **Finger States**: Extension, curl, elevation
5. **Shape Features**: Palm ratios, finger spread
6. **Geometric Features**: Centroids, bounding boxes

## 📈 Model Performance

With real training data, expect:
- **Accuracy**: 85-95%
- **Real-time Performance**: 30+ FPS
- **Robustness**: Handles different users, lighting
- **Generalization**: Works across hand sizes/positions

## 🛠️ Requirements

```bash
pip install opencv-python mediapipe scikit-learn numpy matplotlib seaborn
```

## 🎯 Usage Tips

### For Best Results:
1. **Collect Real Data**: Use the data collector for your specific use case
2. **Vary Conditions**: Different lighting, hand positions, users
3. **Quality Over Quantity**: 40-50 good samples per gesture
4. **Retrain Regularly**: Update model with new data

### Performance Optimization:
- Reduce `max_num_hands` for better performance
- Adjust confidence thresholds based on requirements
- Use GPU acceleration if available
- Consider model quantization for deployment

## 🤝 Contributing

When adding new features:
1. Follow the modular structure
2. Add comprehensive tests
3. Update documentation
4. Use type hints
5. Follow existing code style

## 📝 License

[Add your license information here] 