"""
Enhanced Gesture Recognition Demo
================================

Real-time demo showcasing the enhanced gesture classifier with 
smooth recognition and performance metrics.
"""

import cv2
import time
import numpy as np
from collections import deque
import os
import sys
import logging
from typing import Dict, List, Any

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.gesture_detector import GestureDetector
from core.enhanced_gesture_classifier import EnhancedGestureClassifier

logger = logging.getLogger(__name__)

class EnhancedGestureDemo:
    """Real-time enhanced gesture recognition demo."""
    
    def __init__(self, model_path: str = None):
        """Initialize the enhanced demo."""
        self.gesture_detector = GestureDetector()
        self.gesture_classifier = None
        self.model_path = model_path
        
        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        self.prediction_times = deque(maxlen=30)
        self.gesture_history = deque(maxlen=10)
        
        # Display settings
        self.show_landmarks = True
        self.show_connections = True
        self.show_performance = True
        self.show_confidence = True
        
        # Colors for different gestures
        self.gesture_colors = {
            'neutral': (128, 128, 128),
            'pointing': (0, 255, 255),
            'fist': (0, 0, 255),
            'open_hand': (0, 255, 0),
            'peace_sign': (255, 0, 255),
            'thumbs_up': (255, 255, 0),
            'pinch': (255, 165, 0),
            'ok_sign': (0, 255, 127),
            'rock': (255, 20, 147),
            'gun': (139, 69, 19)
        }
    
    def initialize(self) -> bool:
        """Initialize all components."""
        # Initialize gesture detector
        config = {
            "static_image_mode": False,
            "max_num_hands": 1,
            "min_detection_confidence": 0.8,
            "min_tracking_confidence": 0.7
        }
        
        if not self.gesture_detector.initialize(config):
            print("❌ Failed to initialize gesture detector")
            return False
        
        # Load enhanced model
        if self.model_path is None:
            # Find the latest enhanced model
            self.model_path = self._find_latest_enhanced_model()
        
        if self.model_path is None or not os.path.exists(self.model_path):
            print("❌ Enhanced model not found. Please train a model first:")
            print("   python -m tools.enhanced_trainer")
            return False
        
        # Initialize classifier
        self.gesture_classifier = EnhancedGestureClassifier()
        
        try:
            self.gesture_classifier.load_enhanced_model(self.model_path)
            print(f"✅ Enhanced model loaded: {self.model_path}")
            
            # Display model info
            model_info = self.gesture_classifier.get_model_info()
            print(f"📊 Model type: {model_info['model_type']}")
            print(f"🎭 Gestures: {len(model_info['supported_gestures'])}")
            print(f"⚡ Temporal smoothing: {model_info['use_temporal']}")
            print(f"🎯 Ensemble: {model_info['use_ensemble']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load enhanced model: {e}")
            return False
    
    def _find_latest_enhanced_model(self) -> str:
        """Find the latest enhanced model file."""
        data_dir = "data/gesture_data"
        if not os.path.exists(data_dir):
            return None
        
        model_files = [f for f in os.listdir(data_dir) if f.startswith('enhanced_model_') and f.endswith('.pkl')]
        
        if not model_files:
            return None
        
        # Sort by timestamp (newest first)
        model_files.sort(reverse=True)
        return os.path.join(data_dir, model_files[0])
    
    def run_demo(self):
        """Run the enhanced gesture recognition demo."""
        print("\n🚀 Enhanced Gesture Recognition Demo")
        print("=" * 50)
        print("Controls:")
        print("  L - Toggle landmarks")
        print("  C - Toggle connections")
        print("  P - Toggle performance metrics")
        print("  F - Toggle confidence display")
        print("  Q/ESC - Quit")
        print("\nStarting camera...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("✅ Camera initialized")
        print("🎬 Demo running... (press Q to quit)")
        
        frame_count = 0
        last_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                current_time = time.time()
                
                # FPS calculation
                if frame_count > 0:
                    fps = 1.0 / (current_time - last_time)
                    self.fps_buffer.append(fps)
                
                last_time = current_time
                frame_count += 1
                
                # Detect hands
                detection_start = time.time()
                results = self.gesture_detector.detect_hands(frame)
                
                # Enhanced gesture recognition
                gesture_result = None
                if results.get("hands_detected"):
                    hand_data = results["hands"][0]
                    landmarks = hand_data["landmarks"]
                    
                    # Predict gesture with enhanced classifier
                    prediction_start = time.time()
                    gesture_result = self.gesture_classifier.predict_gesture(landmarks)
                    prediction_time = (time.time() - prediction_start) * 1000  # ms
                    
                    self.prediction_times.append(prediction_time)
                    
                    # Store in history for smoothness
                    if gesture_result['confidence'] > 0.6:  # Only store high-confidence predictions
                        self.gesture_history.append(gesture_result['gesture'])
                
                # Draw enhanced UI
                display_frame = self._draw_enhanced_ui(frame, results, gesture_result)
                
                # Show frame
                cv2.imshow('Enhanced Gesture Recognition Demo', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('l'):
                    self.show_landmarks = not self.show_landmarks
                    print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
                elif key == ord('c'):
                    self.show_connections = not self.show_connections
                    print(f"Connections: {'ON' if self.show_connections else 'OFF'}")
                elif key == ord('p'):
                    self.show_performance = not self.show_performance
                    print(f"Performance metrics: {'ON' if self.show_performance else 'OFF'}")
                elif key == ord('f'):
                    self.show_confidence = not self.show_confidence
                    print(f"Confidence display: {'ON' if self.show_confidence else 'OFF'}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Show final statistics
            self._show_final_stats()
    
    def _draw_enhanced_ui(self, frame: np.ndarray, results: Dict, gesture_result: Dict = None) -> np.ndarray:
        """Draw enhanced UI with all information."""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Draw hand detection results
        if results.get("hands_detected"):
            hand_data = results["hands"][0]
            landmarks = hand_data["landmarks"]
            bbox = hand_data["bounding_box"]
            
            # Draw landmarks
            if self.show_landmarks:
                for landmark in landmarks:
                    cv2.circle(display_frame, (landmark["x"], landmark["y"]), 4, (0, 255, 0), -1)
                    cv2.circle(display_frame, (landmark["x"], landmark["y"]), 6, (255, 255, 255), 1)
            
            # Draw connections
            if self.show_connections:
                display_frame = self.gesture_detector.draw_connections(display_frame, results)
            
            # Draw bounding box with gesture-specific color
            gesture_name = gesture_result['gesture'] if gesture_result else 'neutral'
            bbox_color = self.gesture_colors.get(gesture_name, (255, 255, 255))
            
            cv2.rectangle(display_frame,
                        (bbox["x"], bbox["y"]),
                        (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
                        bbox_color, 3)
            
            # Draw gesture prediction
            if gesture_result:
                self._draw_gesture_info(display_frame, gesture_result, bbox)
        
        else:
            # No hand detected
            cv2.putText(display_frame, "No hand detected", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw performance metrics
        if self.show_performance:
            self._draw_performance_metrics(display_frame)
        
        # Draw gesture history
        self._draw_gesture_history(display_frame)
        
        # Draw controls
        self._draw_controls(display_frame)
        
        return display_frame
    
    def _draw_gesture_info(self, frame: np.ndarray, gesture_result: Dict, bbox: Dict):
        """Draw gesture prediction information."""
        gesture = gesture_result['gesture']
        confidence = gesture_result['confidence']
        
        # Main gesture display
        gesture_color = self.gesture_colors.get(gesture, (255, 255, 255))
        
        # Large gesture name
        font_scale = 2.0
        thickness = 3
        gesture_text = gesture.upper().replace('_', ' ')
        
        # Calculate text size and position
        (text_width, text_height), _ = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        x = bbox["x"]
        y = max(bbox["y"] - 10, text_height + 10)
        
        # Draw text background
        cv2.rectangle(frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), (0, 0, 0), -1)
        
        # Draw gesture name
        cv2.putText(frame, gesture_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, gesture_color, thickness)
        
        # Draw confidence if enabled
        if self.show_confidence:
            confidence_text = f"{confidence:.2%}"
            cv2.putText(frame, confidence_text, (x, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Confidence bar
            bar_width = 150
            bar_height = 10
            bar_x = x
            bar_y = y + 50
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            
            # Confidence bar
            conf_width = int(bar_width * confidence)
            bar_color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 0, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), bar_color, -1)
        
        # Draw probabilities for top 3 gestures
        if 'probabilities' in gesture_result:
            probs = gesture_result['probabilities']
            sorted_gestures = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for i, (gest, prob) in enumerate(sorted_gestures):
                prob_text = f"{gest}: {prob:.2f}"
                y_offset = y + 80 + i * 25
                color = gesture_color if gest == gesture else (200, 200, 200)
                cv2.putText(frame, prob_text, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _draw_performance_metrics(self, frame: np.ndarray):
        """Draw real-time performance metrics."""
        h, w = frame.shape[:2]
        
        # Calculate metrics
        avg_fps = np.mean(self.fps_buffer) if self.fps_buffer else 0
        avg_pred_time = np.mean(self.prediction_times) if self.prediction_times else 0
        
        # Performance box background
        box_x = w - 250
        box_y = 10
        box_w = 240
        box_h = 100
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Performance text
        metrics = [
            f"FPS: {avg_fps:.1f}",
            f"Pred Time: {avg_pred_time:.1f}ms",
            f"Buffer: {len(self.gesture_classifier.temporal_buffer) if self.gesture_classifier else 0}"
        ]
        
        for i, metric in enumerate(metrics):
            cv2.putText(frame, metric, (box_x + 10, box_y + 25 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Performance indicator
        if avg_fps > 30:
            perf_color = (0, 255, 0)  # Green
            perf_text = "EXCELLENT"
        elif avg_fps > 15:
            perf_color = (0, 255, 255)  # Yellow
            perf_text = "GOOD"
        else:
            perf_color = (0, 0, 255)  # Red
            perf_text = "POOR"
        
        cv2.circle(frame, (box_x + box_w - 20, box_y + 20), 8, perf_color, -1)
        cv2.putText(frame, perf_text, (box_x + 10, box_y + box_h - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, perf_color, 2)
    
    def _draw_gesture_history(self, frame: np.ndarray):
        """Draw recent gesture history."""
        if not self.gesture_history:
            return
        
        h, w = frame.shape[:2]
        
        # History display
        history_text = " → ".join(list(self.gesture_history)[-5:])  # Last 5 gestures
        cv2.putText(frame, f"History: {history_text}", (20, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _draw_controls(self, frame: np.ndarray):
        """Draw control instructions."""
        h, w = frame.shape[:2]
        
        controls = ["L:Landmarks", "C:Connections", "P:Performance", "F:Confidence", "Q:Quit"]
        control_text = "  ".join(controls)
        
        cv2.putText(frame, control_text, (20, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def _show_final_stats(self):
        """Show final performance statistics."""
        print("\n📊 Session Statistics")
        print("=" * 30)
        
        if self.fps_buffer:
            avg_fps = np.mean(self.fps_buffer)
            min_fps = np.min(self.fps_buffer)
            max_fps = np.max(self.fps_buffer)
            print(f"FPS: avg={avg_fps:.1f}, min={min_fps:.1f}, max={max_fps:.1f}")
        
        if self.prediction_times:
            avg_pred = np.mean(self.prediction_times)
            min_pred = np.min(self.prediction_times)
            max_pred = np.max(self.prediction_times)
            print(f"Prediction Time: avg={avg_pred:.1f}ms, min={min_pred:.1f}ms, max={max_pred:.1f}ms")
        
        if self.gesture_history:
            unique_gestures = set(self.gesture_history)
            print(f"Gestures detected: {len(unique_gestures)}")
            print(f"Most common: {max(set(self.gesture_history), key=list(self.gesture_history).count)}")

def main():
    """Main demo function."""
    demo = EnhancedGestureDemo()
    
    if not demo.initialize():
        return
    
    demo.run_demo()

if __name__ == "__main__":
    main() 