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
    
    def detect_gestures(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect hand gestures in the given frame.
        
        Args:
            frame (np.ndarray): Input camera frame
            
        Returns:
            Dict[str, Any]: Detection results
        """
        if not self.is_initialized:
            return {"error": "Gesture detector not initialized"}
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        detection_results = {
            "hands_detected": False,
            "hand_landmarks": [],
            "gestures": [],
            "selection_area": None
        }
        
        if results.multi_hand_landmarks:
            detection_results["hands_detected"] = True
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmark coordinates
                landmarks = self._extract_landmarks(hand_landmarks)
                detection_results["hand_landmarks"].append(landmarks)
                
                # Detect specific gestures
                gestures = self._classify_gestures(landmarks)
                detection_results["gestures"].extend(gestures)
                
                # Detect selection area
                selection_area = self._detect_selection_area(landmarks)
                if selection_area:
                    detection_results["selection_area"] = selection_area
        
        return detection_results
    
    def _extract_landmarks(self, hand_landmarks) -> List[Tuple[float, float]]:
        """Extract hand landmark coordinates."""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y))
        return landmarks
    
    def _classify_gestures(self, landmarks: List[Tuple[float, float]]) -> List[str]:
        """Classify hand gestures based on landmarks."""
        gestures = []
        
        # Placeholder for gesture classification logic
        # This would contain actual gesture recognition algorithms
        
        # Example: Simple pointing gesture detection
        if self._is_pointing_gesture(landmarks):
            gestures.append("pointing")
        
        # Example: Selection gesture detection
        if self._is_selection_gesture(landmarks):
            gestures.append("selection")
        
        return gestures
    
    def _is_pointing_gesture(self, landmarks: List[Tuple[float, float]]) -> bool:
        """Detect pointing gesture."""
        # Placeholder implementation
        # In a real implementation, this would analyze finger positions
        return False
    
    def _is_selection_gesture(self, landmarks: List[Tuple[float, float]]) -> bool:
        """Detect selection gesture."""
        # Placeholder implementation
        # In a real implementation, this would analyze hand shape
        return False
    
    def _detect_selection_area(self, landmarks: List[Tuple[float, float]]) -> Optional[Dict[str, float]]:
        """Detect area selection based on hand position."""
        # Placeholder implementation
        # In a real implementation, this would calculate selection boundaries
        return None
    
    def draw_landmarks(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Draw hand landmarks on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            results (Dict[str, Any]): Detection results
            
        Returns:
            np.ndarray: Frame with landmarks drawn
        """
        annotated_frame = frame.copy()
        
        if results.get("hands_detected") and results.get("hand_landmarks"):
            # Draw landmarks for each detected hand
            for landmarks in results["hand_landmarks"]:
                # Convert normalized coordinates to pixel coordinates
                h, w, _ = frame.shape
                for x, y in landmarks:
                    px, py = int(x * w), int(y * h)
                    cv2.circle(annotated_frame, (px, py), 5, (0, 255, 0), -1)
        
        return annotated_frame
    
    def cleanup(self):
        """Clean up resources."""
        if self.hands:
            self.hands.close()
        self.is_initialized = False
        logger.info("Gesture detector cleaned up") 