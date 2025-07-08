"""
Gesture Classification Demo
==========================

Real-time demo combining hand tracking with ML-based gesture classification.
Shows live gesture recognition including pointing, selection, swipes, and more.
"""

import cv2
import time
import os
import logging
from typing import Dict, Any
from ..core.gesture_detector import GestureDetector
from ..core.gesture_classifier import GestureClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GestureRecognitionDemo:
    """Complete gesture recognition demo with hand tracking + ML classification."""
    
    def __init__(self):
        """Initialize the demo with hand tracking and gesture classification."""
        self.hand_detector = GestureDetector()
        self.gesture_classifier = GestureClassifier(model_type='knn', temporal_window=10)
        self.is_initialized = False
        
        # Demo statistics
        self.gesture_counts = {}
        self.total_frames = 0
        self.detection_frames = 0
        
        # Display settings
        self.show_landmarks = True
        self.show_connections = True
        self.show_bbox = True
        self.show_probabilities = False
        
    def initialize(self) -> bool:
        """Initialize both hand tracking and gesture classification."""
        # Initialize hand detector
        hand_config = {
            "static_image_mode": False,
            "max_num_hands": 1,  # Focus on single hand for better performance
            "min_detection_confidence": 0.7,
            "min_tracking_confidence": 0.6
        }
        
        if not self.hand_detector.initialize(hand_config):
            logger.error("Failed to initialize hand detector")
            return False
        
        # Load or train gesture classifier
        model_path = "../data/gestures/optimized_gesture_model.pkl"
        if os.path.exists(model_path):
            logger.info("Loading pre-trained optimized model...")
            self.gesture_classifier.load_model(model_path)
        else:
            logger.info("Training gesture classifier with real data...")
            data_file = "../data/gestures/training_data.json"
            self.gesture_classifier.train(data_file=data_file)
        
        self.is_initialized = True
        logger.info("Demo initialized successfully")
        return True
    
    def process_frame(self, frame) -> Dict[str, Any]:
        """Process a frame for both hand detection and gesture classification."""
        self.total_frames += 1
        
        # Detect hands
        hand_results = self.hand_detector.detect_hands(frame)
        
        results = {
            "hands_detected": hand_results.get("hands_detected", False),
            "gesture_results": [],
            "hand_data": hand_results.get("hands", [])
        }
        
        if hand_results.get("hands_detected"):
            self.detection_frames += 1
            
            for hand_data in hand_results["hands"]:
                landmarks = hand_data["landmarks"]
                
                # Classify gesture
                gesture_result = self.gesture_classifier.predict_gesture(landmarks)
                
                # Update statistics
                gesture_name = gesture_result.get("gesture", "unknown")
                self.gesture_counts[gesture_name] = self.gesture_counts.get(gesture_name, 0) + 1
                
                # Combine results
                combined_result = {
                    "handedness": hand_data["handedness"],
                    "bounding_box": hand_data["bounding_box"],
                    "landmarks": landmarks,
                    "gesture": gesture_result
                }
                
                results["gesture_results"].append(combined_result)
        
        return results
    
    def draw_results(self, frame, results: Dict[str, Any]):
        """Draw all detection and classification results on the frame."""
        annotated_frame = frame.copy()
        
        if not results["hands_detected"]:
            # Draw "No hands detected" message
            cv2.putText(annotated_frame, "No hands detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return annotated_frame
        
        for result in results["gesture_results"]:
            handedness = result["handedness"]
            bbox = result["bounding_box"]
            landmarks = result["landmarks"]
            gesture_info = result["gesture"]
            
            # Draw bounding box
            if self.show_bbox:
                color = (0, 255, 0) if handedness["label"] == "Right" else (255, 0, 0)
                cv2.rectangle(annotated_frame,
                            (bbox["x"], bbox["y"]),
                            (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
                            color, 2)
            
            # Draw landmarks
            if self.show_landmarks:
                for landmark in landmarks:
                    cv2.circle(annotated_frame, (landmark["x"], landmark["y"]), 3, (0, 255, 255), -1)
            
            # Draw hand connections
            if self.show_connections:
                self._draw_hand_skeleton(annotated_frame, landmarks)
            
            # Draw gesture information
            self._draw_gesture_info(annotated_frame, bbox, handedness, gesture_info)
            
            # Draw probabilities if enabled
            if self.show_probabilities and gesture_info.get("probabilities"):
                self._draw_probabilities(annotated_frame, gesture_info["probabilities"])
        
        return annotated_frame
    
    def _draw_hand_skeleton(self, frame, landmarks):
        """Draw hand skeleton connections."""
        # Define hand connections (simplified)
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm connections
            (5, 9), (9, 13), (13, 17)
        ]
        
        for start_idx, end_idx in connections:
            start_point = (landmarks[start_idx]["x"], landmarks[start_idx]["y"])
            end_point = (landmarks[end_idx]["x"], landmarks[end_idx]["y"])
            cv2.line(frame, start_point, end_point, (255, 255, 255), 1)
    
    def _draw_gesture_info(self, frame, bbox, handedness, gesture_info):
        """Draw gesture classification information."""
        gesture_name = gesture_info.get("gesture", "unknown")
        confidence = gesture_info.get("confidence", 0.0)
        gesture_type = gesture_info.get("type", "static")
        
        # Main gesture label
        label = f"{handedness['label']}: {gesture_name}"
        confidence_text = f"({confidence:.2f})"
        
        # Color coding for different gesture types
        if gesture_type == "temporal":
            color = (255, 165, 0)  # Orange for temporal gestures (swipes)
        elif confidence > 0.8:
            color = (0, 255, 0)    # Green for high confidence
        elif confidence > 0.5:
            color = (255, 255, 0)  # Yellow for medium confidence
        else:
            color = (255, 0, 0)    # Red for low confidence
        
        # Draw gesture label above bounding box
        label_y = bbox["y"] - 40 if bbox["y"] > 50 else bbox["y"] + bbox["height"] + 30
        cv2.putText(frame, label, (bbox["x"], label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw confidence below gesture name
        cv2.putText(frame, confidence_text, (bbox["x"], label_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Draw gesture type indicator
        type_indicator = f"[{gesture_type}]"
        cv2.putText(frame, type_indicator, (bbox["x"] + bbox["width"] - 80, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_probabilities(self, frame, probabilities: Dict[str, float]):
        """Draw gesture probabilities sidebar."""
        y_start = 100
        x_pos = frame.shape[1] - 200
        
        cv2.putText(frame, "Probabilities:", (x_pos, y_start - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        for i, (gesture, prob) in enumerate(sorted_probs[:5]):  # Top 5
            y_pos = y_start + i * 25
            text = f"{gesture}: {prob:.3f}"
            
            # Color based on probability
            if prob > 0.5:
                color = (0, 255, 0)
            elif prob > 0.2:
                color = (255, 255, 0)
            else:
                color = (128, 128, 128)
            
            cv2.putText(frame, text, (x_pos, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def draw_statistics(self, frame):
        """Draw demo statistics on the frame."""
        stats_y = 30
        
        # Detection rate
        detection_rate = (self.detection_frames / max(self.total_frames, 1)) * 100
        stats_text = f"Detection Rate: {detection_rate:.1f}%"
        cv2.putText(frame, stats_text, (10, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Most common gesture
        if self.gesture_counts:
            most_common = max(self.gesture_counts, key=self.gesture_counts.get)
            count = self.gesture_counts[most_common]
            common_text = f"Most Common: {most_common} ({count})"
            cv2.putText(frame, common_text, (10, stats_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def run_demo(self):
        """Run the main demo loop."""
        if not self.is_initialized:
            logger.error("Demo not initialized")
            return
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\nGesture Recognition Demo")
        print("=======================")
        print("Supported Gestures:")
        gesture_info = self.gesture_classifier.get_gesture_info()
        for gesture in gesture_info["supported_gestures"]:
            print(f"  - {gesture}")
        
        print("\nControls:")
        print("  q - Quit")
        print("  l - Toggle landmarks")
        print("  c - Toggle connections")
        print("  b - Toggle bounding box")
        print("  p - Toggle probabilities")
        print("  r - Reset statistics")
        print()
        
        fps_start_time = time.time()
        frame_count = 0
        fps = 0.0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Could not read frame")
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                results = self.process_frame(frame)
                
                # Draw results
                display_frame = self.draw_results(frame, results)
                
                # Draw statistics
                self.draw_statistics(display_frame)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps_end_time = time.time()
                    elapsed_time = fps_end_time - fps_start_time
                    if elapsed_time > 0:
                        fps = 30 / elapsed_time
                    fps_start_time = fps_end_time
                
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Display frame
                cv2.imshow('Gesture Recognition Demo', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('l'):
                    self.show_landmarks = not self.show_landmarks
                    print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
                elif key == ord('c'):
                    self.show_connections = not self.show_connections
                    print(f"Connections: {'ON' if self.show_connections else 'OFF'}")
                elif key == ord('b'):
                    self.show_bbox = not self.show_bbox
                    print(f"Bounding Box: {'ON' if self.show_bbox else 'OFF'}")
                elif key == ord('p'):
                    self.show_probabilities = not self.show_probabilities
                    print(f"Probabilities: {'ON' if self.show_probabilities else 'OFF'}")
                elif key == ord('r'):
                    self.gesture_counts.clear()
                    self.total_frames = 0
                    self.detection_frames = 0
                    print("Statistics reset")
        
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.hand_detector.cleanup()
            
            # Print final statistics
            self._print_final_stats()
    
    def _print_final_stats(self):
        """Print final demo statistics."""
        print("\n" + "="*50)
        print("DEMO STATISTICS")
        print("="*50)
        print(f"Total frames processed: {self.total_frames}")
        print(f"Frames with hand detection: {self.detection_frames}")
        
        if self.total_frames > 0:
            detection_rate = (self.detection_frames / self.total_frames) * 100
            print(f"Detection rate: {detection_rate:.1f}%")
        
        if self.gesture_counts:
            print("\nGesture counts:")
            sorted_gestures = sorted(self.gesture_counts.items(), key=lambda x: x[1], reverse=True)
            for gesture, count in sorted_gestures:
                percentage = (count / sum(self.gesture_counts.values())) * 100
                print(f"  {gesture}: {count} ({percentage:.1f}%)")

def main():
    """Main function to run the gesture recognition demo."""
    demo = GestureRecognitionDemo()
    
    if demo.initialize():
        demo.run_demo()
    else:
        logger.error("Failed to initialize demo")

if __name__ == "__main__":
    main() 