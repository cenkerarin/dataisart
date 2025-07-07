"""
Hand Tracking Demo
=================

Demonstration script for real-time hand tracking using the GestureDetector module.
Shows how to extract hand landmarks, handedness, and bounding boxes.
"""

import cv2
import time
import json
from gesture_detector import GestureDetector

def main():
    """Main demo function for real-time hand tracking."""
    
    # Initialize the gesture detector
    detector = GestureDetector()
    
    # Configuration for hand tracking
    config = {
        "static_image_mode": False,
        "max_num_hands": 2,
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.5
    }
    
    # Initialize the detector
    if not detector.initialize(config):
        print("Failed to initialize gesture detector")
        return
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use default camera
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Hand Tracking Demo")
    print("==================")
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save current hand data to JSON")
    print("- Press 'l' to toggle landmark display")
    print("- Press 'c' to toggle connection display")
    print()
    
    # Demo state
    show_landmarks = True
    show_connections = True
    frame_count = 0
    fps = 0.0  # Initialize FPS variable
    fps_start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            results = detector.detect_hands(frame)
            
            # Display results
            if results.get("hands_detected"):
                print(f"\rHands detected: {len(results['hands'])}", end="")
                
                # Display hand information
                for i, hand_data in enumerate(results["hands"]):
                    handedness = hand_data["handedness"]
                    landmarks = hand_data["landmarks"]
                    bbox = hand_data["bounding_box"]
                    
                    print(f"\nHand {i+1}: {handedness['label']} (confidence: {handedness['score']:.2f})")
                    print(f"  Bounding box: x={bbox['x']}, y={bbox['y']}, w={bbox['width']}, h={bbox['height']}")
                    print(f"  Landmarks: {len(landmarks)} points detected")
            else:
                print(f"\rNo hands detected", end="")
            
            # Draw annotations
            display_frame = frame.copy()
            
            if show_connections:
                display_frame = detector.draw_connections(display_frame, results)
            
            if show_landmarks:
                display_frame = detector.draw_landmarks(display_frame, results)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:  # Update FPS every 30 frames
                fps_end_time = time.time()
                elapsed_time = fps_end_time - fps_start_time
                if elapsed_time > 0:  # Avoid division by zero
                    fps = 30 / elapsed_time
                fps_start_time = fps_end_time
                
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Hand Tracking Demo', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current hand data to JSON
                if results.get("hands_detected"):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"hand_data_{timestamp}.json"
                    with open(filename, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"\nHand data saved to {filename}")
                else:
                    print("\nNo hands detected to save")
            elif key == ord('l'):
                show_landmarks = not show_landmarks
                print(f"\nLandmark display: {'ON' if show_landmarks else 'OFF'}")
            elif key == ord('c'):
                show_connections = not show_connections
                print(f"\nConnection display: {'ON' if show_connections else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        detector.cleanup()
        print("\nDemo ended")

def print_hand_info(hand_data: dict):
    """Print detailed information about detected hand."""
    print(f"\nHand Information:")
    print(f"  Handedness: {hand_data['handedness']['label']} (confidence: {hand_data['handedness']['score']:.3f})")
    
    bbox = hand_data['bounding_box']
    print(f"  Bounding Box: ({bbox['x']}, {bbox['y']}) size: {bbox['width']}x{bbox['height']}")
    
    print(f"  Landmarks (21 key points):")
    landmark_names = [
        "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
    ]
    
    for landmark in hand_data['landmarks']:
        name = landmark_names[landmark['id']] if landmark['id'] < len(landmark_names) else f"Point_{landmark['id']}"
        print(f"    {landmark['id']:2d} {name:20}: ({landmark['x']:4d}, {landmark['y']:4d}) z={landmark['z']:6.3f}")

def save_example_data():
    """Save example hand tracking data structure to JSON for reference."""
    example_data = {
        "hands_detected": True,
        "hands": [
            {
                "landmarks": [
                    {"id": 0, "x": 640, "y": 360, "z": 0.0, "visibility": 1.0},
                    # ... (would contain all 21 landmarks)
                ],
                "landmarks_normalized": [
                    {"id": 0, "x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0},
                    # ... (would contain all 21 landmarks in 0-1 range)
                ],
                "handedness": {
                    "label": "Right",
                    "score": 0.9876
                },
                "bounding_box": {
                    "x": 580,
                    "y": 300,
                    "width": 120,
                    "height": 140
                }
            }
        ],
        "frame_shape": [720, 1280, 3]
    }
    
    with open("example_hand_data_structure.json", "w") as f:
        json.dump(example_data, f, indent=2)
    
    print("Example data structure saved to 'example_hand_data_structure.json'")

if __name__ == "__main__":
    # Save example data structure for reference
    save_example_data()
    
    # Run the main demo
    main() 