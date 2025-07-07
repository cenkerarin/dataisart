"""
Gesture Detector
===============

Handles hand tracking and gesture recognition using MediaPipe.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class GestureDetector:
    """Detects and interprets hand gestures for data interaction."""
    
    def __init__(self):
        """Initialize the gesture detector."""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = None
        self.is_initialized = False
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize MediaPipe hands detection.
        
        Args:
            config (Dict[str, Any]): Hand tracking configuration
            
        Returns:
            bool: True if successful
        """
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=config.get("static_image_mode", False),
                max_num_hands=config.get("max_num_hands", 2),
                min_detection_confidence=config.get("min_detection_confidence", 0.7),
                min_tracking_confidence=config.get("min_tracking_confidence", 0.5)
            )
            self.is_initialized = True
            logger.info("Gesture detector initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing gesture detector: {str(e)}")
            return False
    
    def detect_hands(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect hand landmarks, handedness, and bounding boxes in the given frame.
        
        Args:
            frame (np.ndarray): Input camera frame (BGR format)
            
        Returns:
            Dict[str, Any]: Detection results containing:
                - hands_detected: bool
                - hands: List of hand data with landmarks, handedness, and bounding box
                - frame_shape: tuple of frame dimensions
        """
        if not self.is_initialized:
            return {"error": "Gesture detector not initialized"}
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame.shape
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        detection_results = {
            "hands_detected": False,
            "hands": [],
            "frame_shape": (h, w, c)
        }
        
        if results.multi_hand_landmarks and results.multi_handedness:
            detection_results["hands_detected"] = True
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Extract the 21 key points
                landmarks = self._extract_landmarks(hand_landmarks, w, h)
                
                # Get handedness (left/right)
                hand_label = handedness.classification[0].label
                hand_score = handedness.classification[0].score
                
                # Calculate bounding box
                bounding_box = self._calculate_bounding_box(landmarks)
                
                hand_data = {
                    "landmarks": landmarks,
                    "landmarks_normalized": self._extract_landmarks_normalized(hand_landmarks),
                    "handedness": {
                        "label": hand_label,
                        "score": hand_score
                    },
                    "bounding_box": bounding_box
                }
                
                detection_results["hands"].append(hand_data)
        
        return detection_results
    
    def _extract_landmarks(self, hand_landmarks, frame_width: int, frame_height: int) -> List[Dict[str, float]]:
        """
        Extract hand landmark coordinates in pixel format.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            List[Dict[str, float]]: List of 21 landmarks with x, y, z coordinates
        """
        landmarks = []
        for i, landmark in enumerate(hand_landmarks.landmark):
            landmarks.append({
                "id": i,
                "x": int(landmark.x * frame_width),
                "y": int(landmark.y * frame_height),
                "z": landmark.z,  # Depth information (relative to wrist)
                "visibility": landmark.visibility if hasattr(landmark, 'visibility') else 1.0
            })
        return landmarks
    
    def _extract_landmarks_normalized(self, hand_landmarks) -> List[Dict[str, float]]:
        """
        Extract hand landmark coordinates in normalized format (0-1).
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            List[Dict[str, float]]: List of 21 landmarks with normalized x, y, z coordinates
        """
        landmarks = []
        for i, landmark in enumerate(hand_landmarks.landmark):
            landmarks.append({
                "id": i,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility if hasattr(landmark, 'visibility') else 1.0
            })
        return landmarks
    
    def _calculate_bounding_box(self, landmarks: List[Dict[str, float]]) -> Dict[str, int]:
        """
        Calculate bounding box for the hand based on landmarks.
        
        Args:
            landmarks: List of landmark coordinates
            
        Returns:
            Dict[str, int]: Bounding box with x, y, width, height
        """
        x_coords = [landmark["x"] for landmark in landmarks]
        y_coords = [landmark["y"] for landmark in landmarks]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add some padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        
        return {
            "x": x_min,
            "y": y_min,
            "width": x_max - x_min + 2 * padding,
            "height": y_max - y_min + 2 * padding
        }
    
    def detect_gestures(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility.
        Use detect_hands() for new implementations.
        """
        return self.detect_hands(frame)
    
    def _classify_gestures(self, landmarks: List[Dict[str, float]]) -> List[str]:
        """
        Classify hand gestures based on landmarks.
        This is a placeholder for future gesture classification.
        
        Args:
            landmarks: List of landmark coordinates
            
        Returns:
            List[str]: List of detected gestures
        """
        gestures = []
        
        # Placeholder for gesture classification logic
        # This can be expanded for specific gesture recognition
        
        # Example: Simple pointing gesture detection
        if self._is_pointing_gesture(landmarks):
            gestures.append("pointing")
        
        # Example: Selection gesture detection
        if self._is_selection_gesture(landmarks):
            gestures.append("selection")
        
        return gestures
    
    def _is_pointing_gesture(self, landmarks: List[Dict[str, float]]) -> bool:
        """
        Detect pointing gesture.
        Placeholder implementation - extend based on your gesture requirements.
        """
        # Example: Check if index finger is extended and others are folded
        # This is a simplified example - real implementation would be more sophisticated
        return False
    
    def _is_selection_gesture(self, landmarks: List[Dict[str, float]]) -> bool:
        """
        Detect selection gesture.
        Placeholder implementation - extend based on your gesture requirements.
        """
        # Example: Check for pinch gesture (thumb and index finger close)
        # This is a simplified example - real implementation would be more sophisticated
        return False
    
    def draw_landmarks(self, frame: np.ndarray, detection_results: Dict[str, Any]) -> np.ndarray:
        """
        Draw hand landmarks, handedness, and bounding boxes on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            detection_results (Dict[str, Any]): Detection results from detect_hands()
            
        Returns:
            np.ndarray: Frame with annotations drawn
        """
        annotated_frame = frame.copy()
        
        if detection_results.get("hands_detected") and detection_results.get("hands"):
            for hand_data in detection_results["hands"]:
                landmarks = hand_data["landmarks"]
                handedness = hand_data["handedness"]
                bounding_box = hand_data["bounding_box"]
                
                # Draw landmarks
                for landmark in landmarks:
                    cv2.circle(annotated_frame, (landmark["x"], landmark["y"]), 5, (0, 255, 0), -1)
                    # Optionally draw landmark ID
                    cv2.putText(annotated_frame, str(landmark["id"]), 
                              (landmark["x"] + 5, landmark["y"] - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, 
                            (bounding_box["x"], bounding_box["y"]),
                            (bounding_box["x"] + bounding_box["width"], 
                             bounding_box["y"] + bounding_box["height"]),
                            (255, 0, 0), 2)
                
                # Draw handedness label
                label = f"{handedness['label']} ({handedness['score']:.2f})"
                cv2.putText(annotated_frame, label,
                          (bounding_box["x"], bounding_box["y"] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return annotated_frame
    
    def draw_connections(self, frame: np.ndarray, detection_results: Dict[str, Any]) -> np.ndarray:
        """
        Draw hand connections between landmarks using MediaPipe's drawing utilities.
        
        Args:
            frame (np.ndarray): Input frame
            detection_results (Dict[str, Any]): Detection results from detect_hands()
            
        Returns:
            np.ndarray: Frame with hand connections drawn
        """
        if not detection_results.get("hands_detected"):
            return frame
            
        # Convert back to RGB for MediaPipe drawing, then back to BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # We need to re-run MediaPipe to get the proper landmark objects for drawing
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    rgb_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    def get_landmark_names(self) -> List[str]:
        """
        Get the names of the 21 hand landmarks.
        
        Returns:
            List[str]: Names of landmarks in order (0-20)
        """
        return [
            "WRIST",
            "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
            "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
            "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
            "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
            "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
        ]
    
    def cleanup(self):
        """Clean up resources."""
        if self.hands:
            self.hands.close()
        self.is_initialized = False
        logger.info("Gesture detector cleaned up") 