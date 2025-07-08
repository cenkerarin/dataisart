"""
Real Gesture Data Collector
===========================

Tool for collecting real gesture training data from users.
"""

import cv2
import json
import os
import time
from typing import List, Dict, Any
from ..core.gesture_detector import GestureDetector
import logging

logger = logging.getLogger(__name__)

class GestureDataCollector:
    """Collects real gesture data for training."""
    
    def __init__(self, data_dir: str = "./gestures"):
        """Initialize the data collector."""
        self.data_dir = data_dir
        self.gesture_detector = GestureDetector()
        self.collected_data = {}
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
    def initialize(self) -> bool:
        """Initialize the hand detector."""
        config = {
            "static_image_mode": False,
            "max_num_hands": 1,
            "min_detection_confidence": 0.8,
            "min_tracking_confidence": 0.7
        }
        return self.gesture_detector.initialize(config)
    
    def collect_gesture_samples(self, gesture_name: str, num_samples: int = 50):
        """
        Collect samples for a specific gesture.
        
        Args:
            gesture_name: Name of the gesture to collect
            num_samples: Number of samples to collect
        """
        if gesture_name not in self.collected_data:
            self.collected_data[gesture_name] = []
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"\nCollecting samples for gesture: {gesture_name}")
        print(f"Target: {num_samples} samples")
        print("\nInstructions:")
        print("1. Position your hand in the gesture pose")
        print("2. Press SPACE to capture a sample")
        print("3. Press 'q' to finish early")
        print("4. Try to vary hand position, rotation, and distance")
        print("\nPress any key to start...")
        
        cv2.waitKey(0)
        
        samples_collected = 0
        
        try:
            while samples_collected < num_samples:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                
                # Detect hands
                results = self.gesture_detector.detect_hands(frame)
                
                # Draw results
                display_frame = frame.copy()
                
                if results.get("hands_detected"):
                    hand_data = results["hands"][0]  # Use first hand
                    landmarks = hand_data["landmarks"]
                    bbox = hand_data["bounding_box"]
                    
                    # Draw landmarks and bbox
                    for landmark in landmarks:
                        cv2.circle(display_frame, (landmark["x"], landmark["y"]), 3, (0, 255, 0), -1)
                    
                    cv2.rectangle(display_frame,
                                (bbox["x"], bbox["y"]),
                                (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
                                (0, 255, 0), 2)
                    
                    # Draw ready indicator
                    cv2.putText(display_frame, "READY - Press SPACE to capture",
                              (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "No hand detected",
                              (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Show progress
                progress_text = f"Gesture: {gesture_name} | Samples: {samples_collected}/{num_samples}"
                cv2.putText(display_frame, progress_text,
                          (50, display_frame.shape[0] - 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.imshow('Gesture Data Collection', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' ') and results.get("hands_detected"):
                    # Capture sample
                    hand_data = results["hands"][0]
                    sample = {
                        "gesture": gesture_name,
                        "landmarks": hand_data["landmarks"],
                        "timestamp": time.time(),
                        "sample_id": len(self.collected_data[gesture_name])
                    }
                    
                    self.collected_data[gesture_name].append(sample)
                    samples_collected += 1
                    
                    print(f"Captured sample {samples_collected}/{num_samples}")
                    
                    # Brief pause after capture
                    time.sleep(0.5)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"Collection complete! Collected {samples_collected} samples for {gesture_name}")
    
    def collect_all_gestures(self, gestures: List[str], samples_per_gesture: int = 50):
        """Collect samples for multiple gestures."""
        for gesture in gestures:
            print(f"\n{'='*50}")
            print(f"Collecting data for: {gesture.upper()}")
            print(f"{'='*50}")
            
            input("Press Enter when ready to start collecting...")
            self.collect_gesture_samples(gesture, samples_per_gesture)
            
            if gesture != gestures[-1]:  # Not the last gesture
                input("\nPress Enter to continue to next gesture...")
    
    def save_data(self, filename: str = None):
        """Save collected data to JSON file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"gesture_data_{timestamp}.json"
        
        filepath = os.path.join(self.data_dir, filename)
        
        # Calculate statistics
        total_samples = sum(len(samples) for samples in self.collected_data.values())
        stats = {
            "total_samples": total_samples,
            "gestures": {name: len(samples) for name, samples in self.collected_data.items()},
            "collection_date": time.time()
        }
        
        save_data = {
            "statistics": stats,
            "data": self.collected_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\nData saved to: {filepath}")
        print(f"Total samples: {total_samples}")
        for gesture, count in stats["gestures"].items():
            print(f"  {gesture}: {count} samples")
    
    def load_data(self, filepath: str):
        """Load previously collected data."""
        with open(filepath, 'r') as f:
            save_data = json.load(f)
        
        self.collected_data = save_data["data"]
        stats = save_data["statistics"]
        
        print(f"Loaded data from: {filepath}")
        print(f"Total samples: {stats['total_samples']}")
        for gesture, count in stats["gestures"].items():
            print(f"  {gesture}: {count} samples")

def main():
    """Main function for data collection."""
    collector = GestureDataCollector()
    
    if not collector.initialize():
        print("Failed to initialize hand detector")
        return
    
    # Define gestures to collect
    gestures = [
        "neutral",
        "pointing", 
        "fist",
        "open_hand",
        "peace_sign",
        "thumbs_up",
        "pinch"
    ]
    
    print("Gesture Data Collection Tool")
    print("===========================")
    print("\nThis tool will help you collect real gesture data for training.")
    print("For best results:")
    print("- Vary hand position, rotation, and distance from camera")
    print("- Ensure good lighting")
    print("- Hold each pose steady when capturing")
    print("- Collect samples from different people if possible")
    
    collector.collect_all_gestures(gestures, samples_per_gesture=30)
    collector.save_data()
    
    print("\nData collection complete!")

if __name__ == "__main__":
    main() 