"""
Gesture Classifier
==================

ML-based gesture classification module that takes 21 hand landmarks as input
and classifies them into predefined gestures using scikit-learn.

Supports:
- Pointing (index finger extended)
- Selection (fist or pinch pose)
- Swipe Left/Right (temporal movement analysis)
- Easy addition of new gestures
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import logging

logger = logging.getLogger(__name__)

class GestureClassifier:
    """ML-based gesture classifier for hand landmarks."""
    
    def __init__(self, model_type='knn', temporal_window=10):
        """
        Initialize the gesture classifier.
        
        Args:
            model_type (str): Type of ML model ('knn', 'rf')
            temporal_window (int): Number of frames for temporal analysis
        """
        self.model_type = model_type
        self.temporal_window = temporal_window
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Gesture definitions
        self.gesture_labels = {
            0: 'neutral',
            1: 'pointing',
            2: 'fist',
            3: 'pinch',
            4: 'open_hand',
            5: 'peace_sign',
            6: 'thumbs_up'
        }
        
        # Temporal buffer for swipe detection
        self.landmark_history = deque(maxlen=self.temporal_window)
        self.hand_center_history = deque(maxlen=self.temporal_window)
        
        # Swipe detection parameters
        self.swipe_threshold = 50  # pixels
        self.min_swipe_frames = 5
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on model_type."""
        if self.model_type == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=5, weights='distance')
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'custom':
            # Model will be set externally
            self.model = None
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def extract_features(self, landmarks: List[Dict[str, float]]) -> np.ndarray:
        """
        Extract comprehensive feature vector from hand landmarks.
        
        Args:
            landmarks: List of 21 hand landmarks with x, y, z coordinates
            
        Returns:
            np.ndarray: Feature vector for classification
        """
        if len(landmarks) != 21:
            raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}")
        
        features = []
        
        # Normalize landmarks to be invariant to hand position and size
        normalized_landmarks = self._normalize_landmarks(landmarks)
        
        # 1. Normalized coordinate features (relative to wrist and palm center)
        coord_features = self._extract_coordinate_features(normalized_landmarks)
        features.extend(coord_features)
        
        # 2. Distance features between key points
        distances = self._calculate_enhanced_distances(normalized_landmarks)
        features.extend(distances)
        
        # 3. Angle features between finger segments
        angles = self._calculate_enhanced_angles(normalized_landmarks)
        features.extend(angles)
        
        # 4. Finger state features (extended/bent)
        finger_states = self._calculate_finger_states(normalized_landmarks)
        features.extend(finger_states)
        
        # 5. Hand shape features (palm ratio, finger spread)
        shape_features = self._calculate_shape_features(normalized_landmarks)
        features.extend(shape_features)
        
        # 6. Geometric features (convex hull, centroid)
        geometric_features = self._calculate_geometric_features(normalized_landmarks)
        features.extend(geometric_features)
        
        return np.array(features)
    
    def _normalize_landmarks(self, landmarks: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Normalize landmarks for scale and position invariance."""
        # Calculate hand center (average of all landmarks)
        center_x = sum(lm['x'] for lm in landmarks) / len(landmarks)
        center_y = sum(lm['y'] for lm in landmarks) / len(landmarks)
        center_z = sum(lm['z'] for lm in landmarks) / len(landmarks)
        
        # Calculate hand scale (distance from wrist to middle finger tip)
        wrist = landmarks[0]
        middle_tip = landmarks[12]
        hand_scale = np.sqrt((middle_tip['x'] - wrist['x'])**2 + 
                           (middle_tip['y'] - wrist['y'])**2 + 
                           (middle_tip['z'] - wrist['z'])**2)
        
        if hand_scale == 0:
            hand_scale = 1.0  # Avoid division by zero
        
        # Normalize each landmark
        normalized = []
        for lm in landmarks:
            normalized.append({
                'id': lm['id'],
                'x': (lm['x'] - center_x) / hand_scale,
                'y': (lm['y'] - center_y) / hand_scale,
                'z': (lm['z'] - center_z) / hand_scale,
                'visibility': lm.get('visibility', 1.0)
            })
        
        return normalized
    
    def _extract_coordinate_features(self, landmarks: List[Dict[str, float]]) -> List[float]:
        """Extract coordinate-based features."""
        features = []
        
        # Wrist-relative coordinates for key landmarks
        wrist = landmarks[0]
        key_landmarks = [4, 8, 12, 16, 20]  # Finger tips
        
        for lm_id in key_landmarks:
            lm = landmarks[lm_id]
            features.extend([
                lm['x'] - wrist['x'],
                lm['y'] - wrist['y'],
                lm['z'] - wrist['z']
            ])
        
        return features
    
    def _calculate_enhanced_distances(self, landmarks: List[Dict[str, float]]) -> List[float]:
        """Calculate enhanced distance features."""
        distances = []
        
        # Finger tip to wrist distances
        finger_tips = [4, 8, 12, 16, 20]
        wrist = landmarks[0]
        
        for tip_id in finger_tips:
            tip = landmarks[tip_id]
            dist = np.sqrt((tip['x'] - wrist['x'])**2 + 
                          (tip['y'] - wrist['y'])**2 + 
                          (tip['z'] - wrist['z'])**2)
            distances.append(dist)
        
        # Inter-finger distances
        for i in range(len(finger_tips)):
            for j in range(i + 1, len(finger_tips)):
                tip1 = landmarks[finger_tips[i]]
                tip2 = landmarks[finger_tips[j]]
                dist = np.sqrt((tip1['x'] - tip2['x'])**2 + 
                              (tip1['y'] - tip2['y'])**2 + 
                              (tip1['z'] - tip2['z'])**2)
                distances.append(dist)
        
        # Thumb-finger pinch distances
        thumb_tip = landmarks[4]
        for tip_id in [8, 12, 16, 20]:
            finger_tip = landmarks[tip_id]
            dist = np.sqrt((thumb_tip['x'] - finger_tip['x'])**2 + 
                          (thumb_tip['y'] - finger_tip['y'])**2 + 
                          (thumb_tip['z'] - finger_tip['z'])**2)
            distances.append(dist)
        
        return distances
    
    def _calculate_enhanced_angles(self, landmarks: List[Dict[str, float]]) -> List[float]:
        """Calculate enhanced angle features."""
        angles = []
        
        # Finger bend angles
        finger_joints = [
            [1, 2, 3, 4],    # Thumb
            [5, 6, 7, 8],    # Index
            [9, 10, 11, 12], # Middle
            [13, 14, 15, 16], # Ring
            [17, 18, 19, 20] # Pinky
        ]
        
        for joints in finger_joints:
            for i in range(len(joints) - 2):
                p1 = landmarks[joints[i]]
                p2 = landmarks[joints[i + 1]]
                p3 = landmarks[joints[i + 2]]
                angle = self._calculate_angle_between_points(p1, p2, p3)
                angles.append(angle)
        
        # Inter-finger angles (finger spread)
        finger_mcps = [5, 9, 13, 17]  # MCP joints
        wrist = landmarks[0]
        
        for i in range(len(finger_mcps) - 1):
            mcp1 = landmarks[finger_mcps[i]]
            mcp2 = landmarks[finger_mcps[i + 1]]
            
            # Vectors from wrist to each MCP
            v1 = [mcp1['x'] - wrist['x'], mcp1['y'] - wrist['y']]
            v2 = [mcp2['x'] - wrist['x'], mcp2['y'] - wrist['y']]
            
            # Angle between vectors
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))
                angles.append(angle)
        
        return angles
    
    def _calculate_finger_states(self, landmarks: List[Dict[str, float]]) -> List[float]:
        """Calculate detailed finger state features."""
        states = []
        
        # For each finger, calculate multiple state indicators
        finger_data = [
            ([1, 2, 3, 4], 'thumb'),
            ([5, 6, 7, 8], 'index'),
            ([9, 10, 11, 12], 'middle'),
            ([13, 14, 15, 16], 'ring'),
            ([17, 18, 19, 20], 'pinky')
        ]
        
        for joints, finger_name in finger_data:
            # Extension ratio
            extension = self._finger_extension_ratio(landmarks, joints)
            states.append(extension)
            
            # Curl ratio (how bent the finger is)
            curl = self._finger_curl_ratio(landmarks, joints)
            states.append(curl)
            
            # Tip elevation (how high the tip is relative to base)
            base = landmarks[joints[0]]
            tip = landmarks[joints[3]]
            elevation = (base['y'] - tip['y']) / (abs(base['y']) + 1e-8)
            states.append(elevation)
        
        return states
    
    def _finger_curl_ratio(self, landmarks: List[Dict], joints: List[int]) -> float:
        """Calculate how curled/bent a finger is."""
        if len(joints) < 4:
            return 0.0
        
        # Calculate angles at each joint
        angles = []
        for i in range(len(joints) - 2):
            p1 = landmarks[joints[i]]
            p2 = landmarks[joints[i + 1]]
            p3 = landmarks[joints[i + 2]]
            angle = self._calculate_angle_between_points(p1, p2, p3)
            angles.append(angle)
        
        # Average curl (lower angle = more bent)
        avg_angle = sum(angles) / len(angles)
        curl_ratio = max(0, (180 - avg_angle) / 180)
        return curl_ratio
    
    def _calculate_shape_features(self, landmarks: List[Dict[str, float]]) -> List[float]:
        """Calculate hand shape features."""
        features = []
        
        # Palm width (distance between thumb base and pinky base)
        thumb_base = landmarks[1]
        pinky_base = landmarks[17]
        palm_width = np.sqrt((thumb_base['x'] - pinky_base['x'])**2 + 
                            (thumb_base['y'] - pinky_base['y'])**2)
        features.append(palm_width)
        
        # Palm length (distance from wrist to middle finger base)
        wrist = landmarks[0]
        middle_base = landmarks[9]
        palm_length = np.sqrt((wrist['x'] - middle_base['x'])**2 + 
                             (wrist['y'] - middle_base['y'])**2)
        features.append(palm_length)
        
        # Aspect ratio
        aspect_ratio = palm_width / (palm_length + 1e-8)
        features.append(aspect_ratio)
        
        # Finger spread (standard deviation of finger tip positions)
        finger_tips = [landmarks[i] for i in [8, 12, 16, 20]]
        tip_x_coords = [tip['x'] for tip in finger_tips]
        tip_y_coords = [tip['y'] for tip in finger_tips]
        
        spread_x = np.std(tip_x_coords)
        spread_y = np.std(tip_y_coords)
        features.extend([spread_x, spread_y])
        
        return features
    
    def _calculate_geometric_features(self, landmarks: List[Dict[str, float]]) -> List[float]:
        """Calculate geometric features of the hand shape."""
        features = []
        
        # Centroid of hand
        centroid_x = sum(lm['x'] for lm in landmarks) / len(landmarks)
        centroid_y = sum(lm['y'] for lm in landmarks) / len(landmarks)
        
        # Distance from wrist to centroid
        wrist = landmarks[0]
        centroid_dist = np.sqrt((wrist['x'] - centroid_x)**2 + (wrist['y'] - centroid_y)**2)
        features.append(centroid_dist)
        
        # Bounding box features
        x_coords = [lm['x'] for lm in landmarks]
        y_coords = [lm['y'] for lm in landmarks]
        
        bbox_width = max(x_coords) - min(x_coords)
        bbox_height = max(y_coords) - min(y_coords)
        bbox_area = bbox_width * bbox_height
        bbox_ratio = bbox_width / (bbox_height + 1e-8)
        
        features.extend([bbox_width, bbox_height, bbox_area, bbox_ratio])
        
        # Variance features (hand "openness")
        x_variance = np.var(x_coords)
        y_variance = np.var(y_coords)
        features.extend([x_variance, y_variance])
        
        return features
    
    def _calculate_distances(self, landmarks: List[Dict[str, float]]) -> List[float]:
        """Calculate important distances between landmarks."""
        distances = []
        
        # Finger tip to wrist distances
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        wrist = landmarks[0]
        
        for tip_id in finger_tips:
            tip = landmarks[tip_id]
            dist = np.sqrt((tip['x'] - wrist['x'])**2 + 
                          (tip['y'] - wrist['y'])**2 + 
                          (tip['z'] - wrist['z'])**2)
            distances.append(dist)
        
        # Thumb to index finger distance (for pinch detection)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        pinch_dist = np.sqrt((thumb_tip['x'] - index_tip['x'])**2 + 
                            (thumb_tip['y'] - index_tip['y'])**2 + 
                            (thumb_tip['z'] - index_tip['z'])**2)
        distances.append(pinch_dist)
        
        return distances
    
    def _calculate_angles(self, landmarks: List[Dict[str, float]]) -> List[float]:
        """Calculate angles between finger segments."""
        angles = []
        
        # Finger joint angles
        finger_joints = [
            [1, 2, 3, 4],    # Thumb
            [5, 6, 7, 8],    # Index
            [9, 10, 11, 12], # Middle
            [13, 14, 15, 16], # Ring
            [17, 18, 19, 20] # Pinky
        ]
        
        for joints in finger_joints:
            for i in range(len(joints) - 2):
                p1 = landmarks[joints[i]]
                p2 = landmarks[joints[i + 1]]
                p3 = landmarks[joints[i + 2]]
                
                angle = self._calculate_angle_between_points(p1, p2, p3)
                angles.append(angle)
        
        return angles
    
    def _calculate_angle_between_points(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """Calculate angle between three points."""
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y'], p1['z'] - p2['z']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y'], p3['z'] - p2['z']])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def _calculate_finger_extensions(self, landmarks: List[Dict[str, float]]) -> List[float]:
        """Calculate finger extension ratios."""
        extensions = []
        
        # For each finger, calculate if it's extended
        finger_data = [
            ([1, 2, 3, 4], 'thumb'),
            ([5, 6, 7, 8], 'index'),
            ([9, 10, 11, 12], 'middle'),
            ([13, 14, 15, 16], 'ring'),
            ([17, 18, 19, 20], 'pinky')
        ]
        
        for joints, finger_name in finger_data:
            if finger_name == 'thumb':
                # Special case for thumb (different orientation)
                extension = self._thumb_extension_ratio(landmarks, joints)
            else:
                extension = self._finger_extension_ratio(landmarks, joints)
            extensions.append(extension)
        
        return extensions
    
    def _finger_extension_ratio(self, landmarks: List[Dict], joints: List[int]) -> float:
        """Calculate extension ratio for a finger."""
        mcp = landmarks[joints[0]]  # Metacarpophalangeal joint
        tip = landmarks[joints[3]]  # Finger tip
        
        # Calculate if tip is above mcp (extended)
        if tip['y'] < mcp['y']:  # Assuming y decreases upward
            return 1.0
        else:
            return 0.0
    
    def _thumb_extension_ratio(self, landmarks: List[Dict], joints: List[int]) -> float:
        """Calculate extension ratio for thumb (different orientation)."""
        cmc = landmarks[joints[0]]  # Carpometacarpal joint
        tip = landmarks[joints[3]]  # Thumb tip
        
        # Thumb extension is more about distance from palm
        distance = np.sqrt((tip['x'] - cmc['x'])**2 + (tip['y'] - cmc['y'])**2)
        return min(distance / 50.0, 1.0)  # Normalize
    
    def load_real_training_data(self, data_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load real training data from collected gesture samples.
        
        Args:
            data_file: Path to JSON file with collected gesture data
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels
        """
        import json
        
        if not os.path.exists(data_file):
            logger.warning(f"Data file not found: {data_file}. Using synthetic data.")
            return self.create_synthetic_training_data()
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        features = []
        labels = []
        
        # Map gesture names to IDs
        gesture_name_to_id = {name: id for id, name in self.gesture_labels.items()}
        
        for gesture_name, samples in data["data"].items():
            if gesture_name not in gesture_name_to_id:
                logger.warning(f"Unknown gesture in data: {gesture_name}")
                continue
            
            gesture_id = gesture_name_to_id[gesture_name]
            
            for sample in samples:
                try:
                    landmarks = sample["landmarks"]
                    feature_vector = self.extract_features(landmarks)
                    features.append(feature_vector)
                    labels.append(gesture_id)
                except Exception as e:
                    logger.warning(f"Error processing sample for {gesture_name}: {e}")
                    continue
        
        if len(features) == 0:
            logger.warning("No valid samples found. Using synthetic data.")
            return self.create_synthetic_training_data()
        
        logger.info(f"Loaded {len(features)} real samples from {len(set(labels))} gesture classes")
        return np.array(features), np.array(labels)
    
    def create_synthetic_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic training data for gesture classification.
        DEPRECATED: Use real data collection instead.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels
        """
        logger.warning("Using synthetic training data. Consider collecting real gesture data for better performance.")
        
        features = []
        labels = []
        
        # Generate synthetic data for each gesture class
        for gesture_id, gesture_name in self.gesture_labels.items():
            for _ in range(100):  # 100 samples per gesture
                synthetic_landmarks = self._generate_synthetic_landmarks(gesture_name)
                feature_vector = self.extract_features(synthetic_landmarks)
                features.append(feature_vector)
                labels.append(gesture_id)
        
        return np.array(features), np.array(labels)
    
    def _generate_synthetic_landmarks(self, gesture_name: str) -> List[Dict[str, float]]:
        """Generate synthetic landmarks for a specific gesture."""
        # Base hand pose (neutral)
        landmarks = []
        
        # Generate 21 landmarks with realistic positions
        for i in range(21):
            landmark = {
                'id': i,
                'x': np.random.normal(100 + i * 20, 10),
                'y': np.random.normal(200 + i * 5, 10),
                'z': np.random.normal(0, 5),
                'visibility': 1.0
            }
            landmarks.append(landmark)
        
        # Modify based on gesture
        if gesture_name == 'pointing':
            # Index finger extended, others bent
            landmarks[8]['y'] -= 50  # Index tip higher
            landmarks[12]['y'] += 20  # Middle tip lower
            landmarks[16]['y'] += 20  # Ring tip lower
            landmarks[20]['y'] += 20  # Pinky tip lower
            
        elif gesture_name == 'fist':
            # All fingers bent
            for tip_id in [4, 8, 12, 16, 20]:
                landmarks[tip_id]['y'] += 30
                
        elif gesture_name == 'pinch':
            # Thumb and index close together
            landmarks[4]['x'] = landmarks[8]['x'] + np.random.normal(0, 5)
            landmarks[4]['y'] = landmarks[8]['y'] + np.random.normal(0, 5)
            
        elif gesture_name == 'open_hand':
            # All fingers extended
            for tip_id in [4, 8, 12, 16, 20]:
                landmarks[tip_id]['y'] -= 40
                
        elif gesture_name == 'peace_sign':
            # Index and middle extended
            landmarks[8]['y'] -= 50  # Index
            landmarks[12]['y'] -= 50  # Middle
            landmarks[16]['y'] += 20  # Ring bent
            landmarks[20]['y'] += 20  # Pinky bent
            
        elif gesture_name == 'thumbs_up':
            # Only thumb extended
            landmarks[4]['y'] -= 60
            for tip_id in [8, 12, 16, 20]:
                landmarks[tip_id]['y'] += 25
        
        return landmarks
    
    def train(self, features: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None, 
              data_file: str = "../data/gestures/gesture_data.json"):
        """
        Train the gesture classifier.
        
        Args:
            features: Training features (if None, tries to load real data)
            labels: Training labels (if None, tries to load real data)
            data_file: Path to real gesture data file
        """
        if features is None or labels is None:
            logger.info("Loading training data...")
            features, labels = self.load_real_training_data(data_file)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model trained with accuracy: {accuracy:.3f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, 
                                        target_names=list(self.gesture_labels.values())))
        
        self.is_trained = True
    
    def predict_gesture(self, landmarks: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Predict gesture from hand landmarks.
        
        Args:
            landmarks: List of 21 hand landmarks
            
        Returns:
            Dict with gesture prediction and confidence
        """
        if not self.is_trained:
            return {"gesture": "unknown", "confidence": 0.0, "error": "Model not trained"}
        
        try:
            # Extract features
            features = self.extract_features(landmarks)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
            
            gesture_name = self.gesture_labels.get(prediction, "unknown")
            
            # Update temporal history for swipe detection
            self._update_temporal_history(landmarks)
            
            # Check for swipe gestures
            swipe_result = self._detect_swipe()
            if swipe_result["detected"]:
                return {
                    "gesture": swipe_result["direction"],
                    "confidence": swipe_result["confidence"],
                    "type": "temporal"
                }
            
            return {
                "gesture": gesture_name,
                "confidence": float(confidence),
                "type": "static",
                "probabilities": {self.gesture_labels[i]: float(prob) 
                                for i, prob in enumerate(probabilities)}
            }
            
        except Exception as e:
            logger.error(f"Error in gesture prediction: {str(e)}")
            return {"gesture": "error", "confidence": 0.0, "error": str(e)}
    
    def _update_temporal_history(self, landmarks: List[Dict[str, float]]):
        """Update temporal history for swipe detection."""
        self.landmark_history.append(landmarks)
        
        # Calculate hand center
        x_coords = [lm['x'] for lm in landmarks]
        y_coords = [lm['y'] for lm in landmarks]
        center = {
            'x': sum(x_coords) / len(x_coords),
            'y': sum(y_coords) / len(y_coords)
        }
        self.hand_center_history.append(center)
    
    def _detect_swipe(self) -> Dict[str, Any]:
        """Detect swipe gestures from temporal data."""
        if len(self.hand_center_history) < self.min_swipe_frames:
            return {"detected": False}
        
        # Calculate movement
        centers = list(self.hand_center_history)
        start_center = centers[0]
        end_center = centers[-1]
        
        dx = end_center['x'] - start_center['x']
        dy = end_center['y'] - start_center['y']
        
        # Check for horizontal swipe
        if abs(dx) > self.swipe_threshold and abs(dx) > abs(dy) * 2:
            direction = "swipe_right" if dx > 0 else "swipe_left"
            confidence = min(abs(dx) / (self.swipe_threshold * 2), 1.0)
            
            # Clear history after detection
            self.hand_center_history.clear()
            
            return {
                "detected": True,
                "direction": direction,
                "confidence": confidence
            }
        
        return {"detected": False}
    
    def add_gesture(self, gesture_name: str, gesture_id: int):
        """
        Add a new gesture class to the classifier.
        
        Args:
            gesture_name: Name of the new gesture
            gesture_id: Unique ID for the gesture
        """
        if gesture_id in self.gesture_labels:
            logger.warning(f"Gesture ID {gesture_id} already exists")
            return
        
        self.gesture_labels[gesture_id] = gesture_name
        logger.info(f"Added new gesture: {gesture_name} (ID: {gesture_id})")
        
        # Note: Model will need to be retrained with new data
        self.is_trained = False
    
    def save_model(self, filepath: str):
        """Save the trained model to file."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'gesture_labels': self.gesture_labels,
            'model_type': self.model_type,
            'temporal_window': self.temporal_window
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.gesture_labels = model_data['gesture_labels']
        self.model_type = model_data['model_type']
        self.temporal_window = model_data['temporal_window']
        
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
    
    def get_gesture_info(self) -> Dict[str, Any]:
        """Get information about supported gestures."""
        return {
            "supported_gestures": list(self.gesture_labels.values()),
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "temporal_window": self.temporal_window,
            "total_gestures": len(self.gesture_labels)
        } 