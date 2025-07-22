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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import logging

logger = logging.getLogger(__name__)

class GestureClassifier:
    """ML-based gesture classifier for hand landmarks."""
    
    def __init__(self, model_type='knn', use_feature_selection=True, n_features=50):
        """
        Initialize the gesture classifier.
        
        Args:
            model_type (str): Type of ML model ('knn', 'rf')
            use_feature_selection (bool): Whether to use feature selection
            n_features (int): Number of features to select (if using feature selection)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=n_features) if use_feature_selection else None
        self.use_feature_selection = use_feature_selection
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
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on model_type."""
        if self.model_type == 'knn':
            # Use optimized parameters for better accuracy
            self.model = KNeighborsClassifier(
                n_neighbors=7,  # Slightly higher k for better generalization
                weights='distance',  # Distance weighting
                metric='minkowski',  # Minkowski distance
                p=2  # Euclidean distance
            )
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def extract_features(self, landmarks: List[Dict[str, float]]) -> np.ndarray:
        """
        Extract feature vector from hand landmarks.
        
        Args:
            landmarks: List of 21 hand landmarks with x, y, z coordinates
            
        Returns:
            np.ndarray: Feature vector for classification
        """
        if len(landmarks) != 21:
            raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}")
        
        features = []
        
        # Basic coordinate features (normalized)
        wrist = landmarks[0]
        for landmark in landmarks:
            # Relative to wrist position
            rel_x = landmark['x'] - wrist['x']
            rel_y = landmark['y'] - wrist['y']
            rel_z = landmark['z'] - wrist['z']
            features.extend([rel_x, rel_y, rel_z])
        
        # Distance features between key points
        distances = self._calculate_distances(landmarks)
        features.extend(distances)
        
        # Angle features
        angles = self._calculate_angles(landmarks)
        features.extend(angles)
        
        # Finger extension features
        finger_extensions = self._calculate_finger_extensions(landmarks)
        features.extend(finger_extensions)
        
        # Additional discriminative features
        advanced_features = self._calculate_advanced_features(landmarks)
        features.extend(advanced_features)
        
        return np.array(features)
    
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
    
    def _calculate_advanced_features(self, landmarks: List[Dict[str, float]]) -> List[float]:
        """Calculate advanced discriminative features for better accuracy."""
        features = []
        
        # 1. Palm area (convex hull of palm landmarks)
        palm_landmarks = [0, 1, 5, 9, 13, 17]  # Key palm points
        palm_area = self._calculate_convex_hull_area([landmarks[i] for i in palm_landmarks])
        features.append(palm_area)
        
        # 2. Finger spread angles (between adjacent fingers)
        finger_bases = [5, 9, 13, 17]  # Index, Middle, Ring, Pinky bases
        for i in range(len(finger_bases) - 1):
            angle = self._calculate_angle_between_points(
                landmarks[finger_bases[i]], 
                landmarks[0],  # wrist
                landmarks[finger_bases[i + 1]]
            )
            features.append(angle)
        
        # 3. Finger length ratios
        finger_lengths = []
        finger_segments = [
            [5, 6, 7, 8],    # Index
            [9, 10, 11, 12], # Middle
            [13, 14, 15, 16], # Ring
            [17, 18, 19, 20] # Pinky
        ]
        
        for segments in finger_segments:
            length = 0
            for i in range(len(segments) - 1):
                p1 = landmarks[segments[i]]
                p2 = landmarks[segments[i + 1]]
                length += np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
            finger_lengths.append(length)
        
        # Add finger length ratios
        for i in range(len(finger_lengths) - 1):
            ratio = finger_lengths[i] / (finger_lengths[i + 1] + 1e-8)
            features.append(ratio)
        
        # 4. Hand orientation (angle of palm)
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        orientation = np.arctan2(middle_mcp['y'] - wrist['y'], middle_mcp['x'] - wrist['x'])
        features.append(orientation)
        
        # 5. Finger curvature (angle between finger segments)
        for segments in finger_segments:
            if len(segments) >= 3:
                curvature = self._calculate_angle_between_points(
                    landmarks[segments[0]], 
                    landmarks[segments[1]], 
                    landmarks[segments[2]]
                )
                features.append(curvature)
        
        # 6. Thumb-finger distances (important for pinch detection)
        thumb_tip = landmarks[4]
        for finger_tip in [8, 12, 16, 20]:  # Other finger tips
            tip = landmarks[finger_tip]
            distance = np.sqrt((thumb_tip['x'] - tip['x'])**2 + 
                              (thumb_tip['y'] - tip['y'])**2)
            features.append(distance)
        
        # 7. Bounding box ratio (width/height)
        x_coords = [lm['x'] for lm in landmarks]
        y_coords = [lm['y'] for lm in landmarks]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        aspect_ratio = width / (height + 1e-8)
        features.append(aspect_ratio)
        
        return features
    
    def _calculate_convex_hull_area(self, points: List[Dict[str, float]]) -> float:
        """Calculate area of convex hull for given points."""
        if len(points) < 3:
            return 0.0
        
        # Simple approximation using shoelace formula
        coords = [(p['x'], p['y']) for p in points]
        n = len(coords)
        area = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            area += coords[i][0] * coords[j][1]
            area -= coords[j][0] * coords[i][1]
        
        return abs(area) / 2.0
    
    def create_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic training data for gesture classification.
        In a real application, you would collect actual gesture data.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels
        """
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
    
    def load_real_training_data(self, data_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load real training data from collected gesture samples.
        
        Args:
            data_file: Path to JSON file with collected gesture data
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels
        """
        import json
        
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
            return self.create_training_data()
        
        logger.info(f"Loaded {len(features)} real samples from {len(set(labels))} gesture classes")
        return np.array(features), np.array(labels)
    
    def train(self, features: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None, 
              data_file: str = "data/gesture_data/training_data.json", 
              use_hyperparameter_tuning: bool = True, use_data_augmentation: bool = True):
        """
        Train the gesture classifier with enhanced accuracy improvements.
        
        Args:
            features: Training features (if None, tries to load real data)
            labels: Training labels (if None, tries to load real data)
            data_file: Path to real gesture data file
            use_hyperparameter_tuning: Whether to use grid search for hyperparameters
            use_data_augmentation: Whether to augment training data
        """
        if features is None or labels is None:
            # Try to load real data first
            if os.path.exists(data_file):
                logger.info(f"Loading real training data from {data_file}...")
                features, labels = self.load_real_training_data(data_file)
            else:
                logger.info("No real training data found. Generating synthetic training data...")
                features, labels = self.create_training_data()
        
        # Data augmentation
        if use_data_augmentation:
            logger.info("Applying data augmentation...")
            features, labels = self._augment_data(features, labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        if self.use_feature_selection and self.feature_selector is not None:
            logger.info("Applying feature selection...")
            X_train_scaled = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_scaled = self.feature_selector.transform(X_test_scaled)
        
        # Hyperparameter tuning
        if use_hyperparameter_tuning and self.model_type == 'knn':
            logger.info("Performing hyperparameter tuning...")
            self._tune_hyperparameters(X_train_scaled, y_train)
        
        # Train model
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Cross-validation mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model trained with test accuracy: {accuracy:.3f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, 
                                        target_names=list(self.gesture_labels.values())))
        
        self.is_trained = True
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray):
        """Perform hyperparameter tuning using grid search."""
        if self.model_type == 'knn':
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['minkowski', 'manhattan', 'chebyshev'],
                'p': [1, 2]  # For minkowski distance
            }
            
            grid_search = GridSearchCV(
                KNeighborsClassifier(),
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Update model with best parameters
            self.model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    def _augment_data(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Augment training data to improve robustness."""
        augmented_features = []
        augmented_labels = []
        
        # Keep original data
        augmented_features.append(features)
        augmented_labels.append(labels)
        
        # Add noise augmentation
        noise_levels = [0.01, 0.02, 0.03]
        for noise_level in noise_levels:
            noise = np.random.normal(0, noise_level, features.shape)
            noisy_features = features + noise
            augmented_features.append(noisy_features)
            augmented_labels.append(labels)
        
        # Add scaling augmentation
        scale_factors = [0.95, 1.05]
        for scale_factor in scale_factors:
            scaled_features = features * scale_factor
            augmented_features.append(scaled_features)
            augmented_labels.append(labels)
        
        # Combine all augmented data
        final_features = np.vstack(augmented_features)
        final_labels = np.hstack(augmented_labels)
        
        logger.info(f"Data augmentation: {len(features)} -> {len(final_features)} samples")
        
        return final_features, final_labels
    
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
            
            # Apply feature selection if enabled
            if self.use_feature_selection and self.feature_selector is not None:
                features_scaled = self.feature_selector.transform(features_scaled)
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
            
            gesture_name = self.gesture_labels.get(prediction, "unknown")
            
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
            "total_gestures": len(self.gesture_labels)
        } 