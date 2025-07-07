# Hand Tracking Module

A comprehensive Python module for real-time hand tracking using MediaPipe. This module provides accurate detection of hand landmarks, handedness classification, and bounding box calculation for gesture recognition applications.

## Features

- **21-point hand landmark detection** - Precise tracking of all hand key points
- **Handedness detection** - Identifies left/right hand with confidence scores
- **Bounding box calculation** - Automatic hand region detection
- **Real-time processing** - Optimized for live video streams
- **Gesture classification ready** - Structured output for downstream gesture recognition
- **Comprehensive visualization** - Built-in drawing utilities for landmarks and connections

## Quick Start

### Basic Usage

```python
from gesture_detector import GestureDetector
import cv2

# Initialize the detector
detector = GestureDetector()
config = {
    "max_num_hands": 2,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5
}
detector.initialize(config)

# Process a frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
results = detector.detect_hands(frame)

# Access the results
if results["hands_detected"]:
    for hand in results["hands"]:
        print(f"Hand: {hand['handedness']['label']}")
        print(f"Landmarks: {len(hand['landmarks'])} points")
        print(f"Bounding box: {hand['bounding_box']}")
```

### Running the Demo

```bash
cd src/core/hand_tracking
python demo_hand_tracking.py
```

**Demo Controls:**
- `q` - Quit the demo
- `s` - Save current hand data to JSON
- `l` - Toggle landmark display
- `c` - Toggle connection display

## Output Format

### Detection Results Structure

```python
{
    "hands_detected": bool,
    "hands": [
        {
            "landmarks": [
                {
                    "id": int,           # Landmark ID (0-20)
                    "x": int,            # Pixel X coordinate
                    "y": int,            # Pixel Y coordinate
                    "z": float,          # Depth (relative to wrist)
                    "visibility": float  # Visibility confidence
                },
                # ... 21 total landmarks
            ],
            "landmarks_normalized": [
                {
                    "id": int,           # Landmark ID (0-20)
                    "x": float,          # Normalized X (0-1)
                    "y": float,          # Normalized Y (0-1)
                    "z": float,          # Depth (relative to wrist)
                    "visibility": float  # Visibility confidence
                },
                # ... 21 total landmarks
            ],
            "handedness": {
                "label": str,        # "Left" or "Right"
                "score": float       # Confidence (0-1)
            },
            "bounding_box": {
                "x": int,            # Top-left X
                "y": int,            # Top-left Y
                "width": int,        # Box width
                "height": int        # Box height
            }
        }
    ],
    "frame_shape": [height, width, channels]
}
```

### Hand Landmarks (21 Key Points)

The 21 landmarks follow MediaPipe's hand model:

| ID | Landmark | Description |
|----|----------|-------------|
| 0 | WRIST | Wrist center |
| 1-4 | THUMB_* | Thumb joints (CMC, MCP, IP, TIP) |
| 5-8 | INDEX_FINGER_* | Index finger joints (MCP, PIP, DIP, TIP) |
| 9-12 | MIDDLE_FINGER_* | Middle finger joints (MCP, PIP, DIP, TIP) |
| 13-16 | RING_FINGER_* | Ring finger joints (MCP, PIP, DIP, TIP) |
| 17-20 | PINKY_* | Pinky joints (MCP, PIP, DIP, TIP) |

## API Reference

### GestureDetector Class

#### `__init__()`
Initialize the gesture detector.

#### `initialize(config: Dict[str, Any]) -> bool`
Initialize MediaPipe hands detection.

**Parameters:**
- `config`: Configuration dictionary
  - `static_image_mode` (bool): Whether to treat input as static images
  - `max_num_hands` (int): Maximum number of hands to detect (1-2)
  - `min_detection_confidence` (float): Minimum confidence for detection
  - `min_tracking_confidence` (float): Minimum confidence for tracking

**Returns:** `True` if successful, `False` otherwise

#### `detect_hands(frame: np.ndarray) -> Dict[str, Any]`
Main detection method. Processes a frame and returns hand tracking results.

**Parameters:**
- `frame`: Input BGR image (numpy array)

**Returns:** Detection results dictionary

#### `draw_landmarks(frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray`
Draw hand landmarks and annotations on the frame.

#### `draw_connections(frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray`
Draw hand skeleton connections using MediaPipe's drawing utilities.

#### `get_landmark_names() -> List[str]`
Get the names of the 21 hand landmarks.

#### `cleanup()`
Clean up MediaPipe resources.

## Configuration Options

### Detection Parameters

```python
config = {
    "static_image_mode": False,        # False for video, True for images
    "max_num_hands": 2,                # 1 or 2 hands
    "min_detection_confidence": 0.7,   # 0.0 - 1.0
    "min_tracking_confidence": 0.5     # 0.0 - 1.0
}
```

### Performance Tuning

- **Higher confidence values** = More accurate but may miss some detections
- **Lower confidence values** = More detections but may have false positives
- **`max_num_hands=1`** = Better performance if only tracking one hand
- **`static_image_mode=True`** = Better for single images, slower for video

## Gesture Classification Examples

The module includes basic gesture classification examples in `usage_example.py`:

```python
# Count extended fingers
fingers_extended = count_extended_fingers(landmarks)

# Detect common gestures
if fingers_extended == 1:
    gesture = "pointing"
elif fingers_extended == 0:
    gesture = "fist"
elif fingers_extended == 5:
    gesture = "open_hand"
```

## Use Cases

- **Gesture Recognition**: Use landmarks for custom gesture classification
- **Human-Computer Interaction**: Control interfaces with hand movements
- **Sign Language Recognition**: Foundation for sign language interpretation
- **Augmented Reality**: Hand tracking for AR applications
- **Accessibility Tools**: Alternative input methods
- **Motion Analysis**: Study hand movement patterns

## Performance Notes

- **Real-time capable**: Typically runs at 30+ FPS on modern hardware
- **Memory efficient**: MediaPipe optimized for mobile and desktop
- **GPU acceleration**: Automatic GPU usage when available
- **Multi-hand support**: Up to 2 hands simultaneously

## Dependencies

All dependencies are included in the main `requirements.txt`:

```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
```

## Troubleshooting

### Common Issues

1. **Camera not detected**: Check camera permissions and availability
2. **Poor tracking**: Ensure good lighting and clear hand visibility
3. **Stuttering performance**: Reduce resolution or detection confidence
4. **Import errors**: Ensure all dependencies are installed

### Performance Optimization

```python
# For better performance:
config = {
    "max_num_hands": 1,                # Track only one hand
    "min_detection_confidence": 0.8,   # Higher confidence
    "min_tracking_confidence": 0.7     # Higher tracking confidence
}

# Reduce frame size if needed
frame = cv2.resize(frame, (640, 480))
```

## Examples

### Simple Detection
```python
python demo_hand_tracking.py
```

### Gesture Classification
```python
python usage_example.py
```

### Custom Integration
See `usage_example.py` for a complete example of building gesture classification on top of the hand tracking module.

## Future Enhancements

- [ ] 3D hand pose estimation
- [ ] Multi-hand gesture recognition
- [ ] Custom gesture training pipeline
- [ ] Hand mesh detection
- [ ] Temporal gesture analysis

## License

This module is part of the DataIsArt project and follows the same licensing terms. 