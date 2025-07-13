"""
Hand Tracking Usage Example
===========================

Simple example showing how to use the GestureDetector module for various applications.
"""

import cv2
import numpy as np
from core.gesture_detector import GestureDetector
from typing import Dict, Any, List

class HandTrackingApp:
    """Example application using hand tracking for gesture classification."""
    
    def __init__(self):
        self.detector = GestureDetector()
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize the hand tracking system."""
        config = {
            "static_image_mode": False,
            "max_num_hands": 2,
            "min_detection_confidence": 0.7,
            "min_tracking_confidence": 0.5
        }
        
        self.is_initialized = self.detector.initialize(config)
        return self.is_initialized
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame and extract hand information.
        
        Args:
            frame: Input frame from camera or video
            
        Returns:
            Dict containing processed hand data
        """
        if not self.is_initialized:
            return {"error": "Hand tracking not initialized"}
        
        # Get hand tracking results
        results = self.detector.detect_hands(frame)
        
        # Process results for application-specific use
        processed_data = {
            "frame_shape": results.get("frame_shape"),
            "hands_count": len(results.get("hands", [])),
            "hands_data": []
        }
        
        if results.get("hands_detected"):
            for hand_data in results["hands"]:
                # Extract key information for each hand
                hand_info = {
                    "handedness": hand_data["handedness"]["label"],
                    "confidence": hand_data["handedness"]["score"],
                    "bounding_box": hand_data["bounding_box"],
                    "landmarks": hand_data["landmarks"],
                    "landmarks_normalized": hand_data["landmarks_normalized"]
                }
                
                # Add custom analysis
                hand_info["gesture_analysis"] = self.analyze_hand_pose(hand_data["landmarks"])
                hand_info["hand_center"] = self.calculate_hand_center(hand_data["landmarks"])
                
                processed_data["hands_data"].append(hand_info)
        
        return processed_data
    
    def analyze_hand_pose(self, landmarks: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Analyze hand pose for gesture classification.
        This is where you would implement your specific gesture recognition logic.
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            Dict containing gesture analysis results
        """
        analysis = {
            "fingers_extended": self.count_extended_fingers(landmarks),
            "hand_openness": self.calculate_hand_openness(landmarks),
            "thumb_position": self.analyze_thumb_position(landmarks),
            "gesture_detected": None
        }
        
        # Simple gesture detection examples
        if analysis["fingers_extended"] == 1:
            analysis["gesture_detected"] = "pointing"
        elif analysis["fingers_extended"] == 0:
            analysis["gesture_detected"] = "fist"
        elif analysis["fingers_extended"] == 5:
            analysis["gesture_detected"] = "open_hand"
        elif analysis["fingers_extended"] == 2:
            analysis["gesture_detected"] = "peace_sign"
        
        return analysis
    
    def count_extended_fingers(self, landmarks: List[Dict[str, float]]) -> int:
        """
        Count how many fingers are extended.
        Uses simple heuristic based on landmark positions.
        """
        # Finger tip and joint landmark IDs
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        finger_joints = [3, 6, 10, 14, 18]  # Corresponding joints
        
        extended_count = 0
        
        for tip_id, joint_id in zip(finger_tips, finger_joints):
            tip = landmarks[tip_id]
            joint = landmarks[joint_id]
            
            # Simple check: if tip is higher (lower y) than joint, finger is extended
            if tip["y"] < joint["y"]:
                extended_count += 1
        
        return extended_count
    
    def calculate_hand_openness(self, landmarks: List[Dict[str, float]]) -> float:
        """
        Calculate how "open" the hand is based on finger spread.
        Returns value between 0 (closed fist) and 1 (fully open).
        """
        # Calculate distances between finger tips
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        wrist = landmarks[0]
        
        total_distance = 0
        for tip_id in finger_tips:
            tip = landmarks[tip_id]
            distance = np.sqrt((tip["x"] - wrist["x"])**2 + (tip["y"] - wrist["y"])**2)
            total_distance += distance
        
        # Normalize (this is a simple approximation)
        avg_distance = total_distance / len(finger_tips)
        openness = min(avg_distance / 200.0, 1.0)  # 200 is an empirical scaling factor
        
        return openness
    
    def analyze_thumb_position(self, landmarks: List[Dict[str, float]]) -> str:
        """Analyze thumb position relative to other fingers."""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Simple thumb position analysis
        if abs(thumb_tip["x"] - index_tip["x"]) < 30 and abs(thumb_tip["y"] - index_tip["y"]) < 30:
            return "pinching"
        elif thumb_tip["x"] < index_tip["x"]:
            return "left"
        else:
            return "right"
    
    def calculate_hand_center(self, landmarks: List[Dict[str, float]]) -> Dict[str, int]:
        """Calculate the center point of the hand."""
        x_coords = [landmark["x"] for landmark in landmarks]
        y_coords = [landmark["y"] for landmark in landmarks]
        
        center_x = int(sum(x_coords) / len(x_coords))
        center_y = int(sum(y_coords) / len(y_coords))
        
        return {"x": center_x, "y": center_y}
    
    def cleanup(self):
        """Clean up resources."""
        if self.detector:
            self.detector.cleanup()

def run_simple_demo():
    """Run a simple demonstration of the hand tracking application."""
    app = HandTrackingApp()
    
    if not app.initialize():
        print("Failed to initialize hand tracking")
        return
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        return
    
    print("Simple Hand Tracking Demo")
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            results = app.process_frame(frame)
            
            # Display results
            if results.get("hands_count", 0) > 0:
                for hand_data in results["hands_data"]:
                    gesture = hand_data["gesture_analysis"]["gesture_detected"]
                    fingers = hand_data["gesture_analysis"]["fingers_extended"]
                    handedness = hand_data["handedness"]
                    
                    print(f"\r{handedness} hand: {fingers} fingers, gesture: {gesture}     ", end="")
                    
                    # Draw on frame
                    bbox = hand_data["bounding_box"]
                    cv2.rectangle(frame, (bbox["x"], bbox["y"]), 
                                (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]), 
                                (0, 255, 0), 2)
                    
                    # Draw gesture label
                    label = f"{handedness}: {gesture or 'none'}"
                    cv2.putText(frame, label, (bbox["x"], bbox["y"] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                print(f"\rNo hands detected     ", end="")
            
            cv2.imshow('Hand Tracking App', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nDemo interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        app.cleanup()
        print("\nDemo ended")

if __name__ == "__main__":
    run_simple_demo() 