"""
Enhanced Gesture Classifier
==========================

Next-generation ML-based gesture classifier with advanced feature engineering,
temporal modeling, and ensemble methods for maximum accuracy.
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Tuple, Any
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from collections import deque
import logging
import json
import time
from scipy import stats
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class EnhancedGestureClassifier:
    """Next-generation gesture classifier with advanced ML techniques."""
    
    def __init__(self, use_ensemble=True, use_temporal=True, temporal_window=10):
        """
        Initialize the enhanced classifier.
        
        Args:
            use_ensemble (bool): Use ensemble methods for better accuracy
            use_temporal (bool): Use temporal sequence modeling
            temporal_window (int): Number of frames for temporal analysis
        """
        self.use_ensemble = use_ensemble
        self.use_temporal = use_temporal
        self.temporal_window = temporal_window
        self.temporal_buffer = deque(maxlen=temporal_window)
        
        # Models
        self.ensemble_model = None
        self.base_models = {}
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_selector = None
        self.pca = None  # Dimensionality reduction
        
        # State
        self.is_trained = False
        self.feature_importance = {}
        self.training_stats = {}
        
        # Enhanced gesture set
        self.gesture_labels = {
            0: 'neutral',
            1: 'pointing',
            2: 'fist',
            3: 'pinch',
            4: 'open_hand',
            5: 'peace_sign',
            6: 'thumbs_up',
            7: 'ok_sign',
            8: 'rock',
            9: 'gun'
        }
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the ensemble of ML models."""
        # Optimized KNN
        knn = KNeighborsClassifier(
            n_neighbors=9,
            weights='distance',
            algorithm='ball_tree',  # Faster for high dimensions
            metric='manhattan'  # Often works better for gesture data
        )
        
        # Optimized Random Forest
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        
        # Gradient Boosting for complex patterns
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        
        # SVM for complex decision boundaries (if dataset is large enough)
        svm = SVC(
            C=10.0,
            kernel='rbf',
            gamma='scale',
            probability=True,  # For ensemble voting
            random_state=42
        )
        
        self.base_models = {
            'knn': knn,
            'rf': rf,
            'gb': gb,
            'svm': svm
        }
        
        if self.use_ensemble:
            # Voting classifier with optimized weights
            self.ensemble_model = VotingClassifier(
                estimators=list(self.base_models.items()),
                voting='soft',  # Use probabilities
                weights=[1, 2, 1.5, 1]  # RF gets more weight (usually most reliable)
            )
    
    def extract_enhanced_features(self, landmarks: List[Dict[str, float]]) -> np.ndarray:
        """
        Extract comprehensive feature set for maximum accuracy.
        
        Args:
            landmarks: List of 21 hand landmarks with x, y, z coordinates
            
        Returns:
            np.ndarray: Enhanced feature vector
        """
        if len(landmarks) != 21:
            raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}")
        
        features = []
        
        # 1. Normalized coordinate features (relative to palm center and wrist)
        wrist = landmarks[0]
        palm_center = self._calculate_palm_center(landmarks)
        
        for landmark in landmarks:
            # Relative to wrist
            rel_x = landmark['x'] - wrist['x']
            rel_y = landmark['y'] - wrist['y']
            rel_z = landmark['z'] - wrist['z']
            
            # Relative to palm center
            palm_rel_x = landmark['x'] - palm_center['x']
            palm_rel_y = landmark['y'] - palm_center['y']
            palm_rel_z = landmark['z'] - palm_center['z']
            
            features.extend([rel_x, rel_y, rel_z, palm_rel_x, palm_rel_y, palm_rel_z])
        
        # 2. Enhanced distance features
        distance_features = self._calculate_enhanced_distances(landmarks)
        features.extend(distance_features)
        
        # 3. Advanced angle features
        angle_features = self._calculate_advanced_angles(landmarks)
        features.extend(angle_features)
        
        # 4. Finger state analysis (more detailed)
        finger_features = self._analyze_finger_states(landmarks)
        features.extend(finger_features)
        
        # 5. Geometric shape descriptors
        shape_features = self._calculate_shape_descriptors(landmarks)
        features.extend(shape_features)
        
        # 6. Statistical features
        statistical_features = self._calculate_statistical_features(landmarks)
        features.extend(statistical_features)
        
        # 7. Topological features
        topological_features = self._calculate_topological_features(landmarks)
        features.extend(topological_features)
        
        return np.array(features)
    
    def _calculate_palm_center(self, landmarks: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate the center of the palm."""
        palm_landmarks = [0, 1, 5, 9, 13, 17]  # Key palm points
        x_coords = [landmarks[i]['x'] for i in palm_landmarks]
        y_coords = [landmarks[i]['y'] for i in palm_landmarks]
        z_coords = [landmarks[i]['z'] for i in palm_landmarks]
        
        return {
            'x': np.mean(x_coords),
            'y': np.mean(y_coords),
            'z': np.mean(z_coords)
        }
    
    def _calculate_enhanced_distances(self, landmarks: List[Dict[str, float]]) -> List[float]:
        """Calculate comprehensive distance features."""
        distances = []
        
        # Fingertip to palm center distances
        palm_center = self._calculate_palm_center(landmarks)
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
        for tip_id in finger_tips:
            tip = landmarks[tip_id]
            dist = euclidean([tip['x'], tip['y'], tip['z']], 
                           [palm_center['x'], palm_center['y'], palm_center['z']])
            distances.append(dist)
        
        # Inter-finger distances (all combinations)
        for i, tip1 in enumerate(finger_tips):
            for tip2 in finger_tips[i+1:]:
                p1 = landmarks[tip1]
                p2 = landmarks[tip2]
                dist = euclidean([p1['x'], p1['y'], p1['z']], 
                               [p2['x'], p2['y'], p2['z']])
                distances.append(dist)
        
        # Finger base to tip distances (finger lengths)
        finger_bases = [1, 5, 9, 13, 17]  # Thumb, Index, Middle, Ring, Pinky bases
        for base_id, tip_id in zip(finger_bases, finger_tips):
            base = landmarks[base_id]
            tip = landmarks[tip_id]
            length = euclidean([base['x'], base['y'], base['z']], 
                             [tip['x'], tip['y'], tip['z']])
            distances.append(length)
        
        # Key gesture-specific distances
        # Pinch distance (thumb-index)
        thumb = landmarks[4]
        index = landmarks[8]
        pinch_dist = euclidean([thumb['x'], thumb['y'], thumb['z']], 
                             [index['x'], index['y'], index['z']])
        distances.append(pinch_dist)
        
        # OK sign distance (thumb-middle)
        middle = landmarks[12]
        ok_dist = euclidean([thumb['x'], thumb['y'], thumb['z']], 
                          [middle['x'], middle['y'], middle['z']])
        distances.append(ok_dist)
        
        return distances
    
    def _calculate_advanced_angles(self, landmarks: List[Dict[str, float]]) -> List[float]:
        """Calculate comprehensive angle features."""
        angles = []
        
        # Finger joint angles (more detailed)
        finger_chains = [
            [0, 1, 2, 3, 4],     # Thumb (including wrist)
            [0, 5, 6, 7, 8],     # Index
            [0, 9, 10, 11, 12],  # Middle
            [0, 13, 14, 15, 16], # Ring
            [0, 17, 18, 19, 20]  # Pinky
        ]
        
        for chain in finger_chains:
            for i in range(len(chain) - 2):
                angle = self._calculate_angle_between_points(
                    landmarks[chain[i]], landmarks[chain[i+1]], landmarks[chain[i+2]]
                )
                angles.append(angle)
        
        # Inter-finger angles
        finger_tips = [4, 8, 12, 16, 20]
        wrist = landmarks[0]
        
        for i in range(len(finger_tips) - 1):
            angle = self._calculate_angle_between_points(
                landmarks[finger_tips[i]], wrist, landmarks[finger_tips[i+1]]
            )
            angles.append(angle)
        
        # Palm orientation angles
        palm_normal = self._calculate_palm_normal(landmarks)
        angles.extend(palm_normal)
        
        return angles
    
    def _calculate_palm_normal(self, landmarks: List[Dict[str, float]]) -> List[float]:
        """Calculate palm normal vector for orientation."""
        # Use three points to define palm plane
        wrist = landmarks[0]
        index_base = landmarks[5]
        pinky_base = landmarks[17]
        
        # Two vectors in palm plane
        v1 = np.array([index_base['x'] - wrist['x'], 
                      index_base['y'] - wrist['y'], 
                      index_base['z'] - wrist['z']])
        
        v2 = np.array([pinky_base['x'] - wrist['x'], 
                      pinky_base['y'] - wrist['y'], 
                      pinky_base['z'] - wrist['z']])
        
        # Normal vector (cross product)
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) > 0:
            normal = normal / np.linalg.norm(normal)
        
        return normal.tolist()
    
    def _analyze_finger_states(self, landmarks: List[Dict[str, float]]) -> List[float]:
        """Analyze detailed finger states and configurations."""
        finger_states = []
        
        # Finger extension ratios (more sophisticated)
        finger_data = [
            ([1, 2, 3, 4], 'thumb'),
            ([5, 6, 7, 8], 'index'),
            ([9, 10, 11, 12], 'middle'),
            ([13, 14, 15, 16], 'ring'),
            ([17, 18, 19, 20], 'pinky')
        ]
        
        for joints, finger_name in finger_data:
            # Extension ratio
            extension = self._calculate_finger_extension_ratio(landmarks, joints, finger_name)
            finger_states.append(extension)
            
            # Curvature
            curvature = self._calculate_finger_curvature(landmarks, joints)
            finger_states.append(curvature)
            
            # Spread angle (relative to palm)
            spread = self._calculate_finger_spread(landmarks, joints)
            finger_states.append(spread)
        
        # Special finger configurations
        # Pointing score
        pointing_score = self._calculate_pointing_score(landmarks)
        finger_states.append(pointing_score)
        
        # Pinch score
        pinch_score = self._calculate_pinch_score(landmarks)
        finger_states.append(pinch_score)
        
        # Fist score
        fist_score = self._calculate_fist_score(landmarks)
        finger_states.append(fist_score)
        
        return finger_states
    
    def _calculate_finger_extension_ratio(self, landmarks: List[Dict], joints: List[int], finger_name: str) -> float:
        """Calculate sophisticated finger extension ratio."""
        if finger_name == 'thumb':
            # Thumb: measure distance from CMC to tip vs folded position
            cmc = landmarks[joints[0]]
            tip = landmarks[joints[3]]
            
            # Expected folded distance (empirically derived)
            folded_dist = 30.0  # pixels
            current_dist = euclidean([cmc['x'], cmc['y']], [tip['x'], tip['y']])
            
            return min(current_dist / folded_dist, 2.0)  # Cap at 2.0
        else:
            # Other fingers: use MCP to tip distance relative to palm
            mcp = landmarks[joints[0]]
            tip = landmarks[joints[3]]
            
            # Distance from MCP to tip
            finger_length = euclidean([mcp['x'], mcp['y']], [tip['x'], tip['y']])
            
            # Expected extended length (based on anatomical ratios)
            expected_length = 80.0  # pixels
            
            return min(finger_length / expected_length, 1.5)
    
    def _calculate_finger_curvature(self, landmarks: List[Dict], joints: List[int]) -> float:
        """Calculate finger curvature."""
        if len(joints) < 4:
            return 0.0
        
        # Calculate angles between segments
        angles = []
        for i in range(len(joints) - 2):
            angle = self._calculate_angle_between_points(
                landmarks[joints[i]], landmarks[joints[i+1]], landmarks[joints[i+2]]
            )
            angles.append(angle)
        
        # Average curvature (deviation from straight)
        straight_angle = 180.0
        curvatures = [abs(angle - straight_angle) for angle in angles]
        return np.mean(curvatures)
    
    def _calculate_finger_spread(self, landmarks: List[Dict], joints: List[int]) -> float:
        """Calculate finger spread angle."""
        if len(joints) < 2:
            return 0.0
        
        finger_vector = np.array([landmarks[joints[-1]]['x'] - landmarks[joints[0]]['x'],
                                 landmarks[joints[-1]]['y'] - landmarks[joints[0]]['y']])
        
        # Reference vector (pointing up)
        reference_vector = np.array([0, -1])
        
        # Calculate angle
        cos_angle = np.dot(finger_vector, reference_vector) / (np.linalg.norm(finger_vector) * np.linalg.norm(reference_vector))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def _calculate_pointing_score(self, landmarks: List[Dict[str, float]]) -> float:
        """Calculate how much the hand looks like a pointing gesture."""
        index_extension = self._calculate_finger_extension_ratio(landmarks, [5, 6, 7, 8], 'index')
        middle_extension = self._calculate_finger_extension_ratio(landmarks, [9, 10, 11, 12], 'middle')
        ring_extension = self._calculate_finger_extension_ratio(landmarks, [13, 14, 15, 16], 'ring')
        pinky_extension = self._calculate_finger_extension_ratio(landmarks, [17, 18, 19, 20], 'pinky')
        
        # Pointing: index extended, others folded
        pointing_score = index_extension - np.mean([middle_extension, ring_extension, pinky_extension])
        return max(0, pointing_score)
    
    def _calculate_pinch_score(self, landmarks: List[Dict[str, float]]) -> float:
        """Calculate how much the hand looks like a pinch gesture."""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Distance between thumb and index tips
        pinch_dist = euclidean([thumb_tip['x'], thumb_tip['y']], 
                              [index_tip['x'], index_tip['y']])
        
        # Smaller distance = higher pinch score
        max_pinch_dist = 50.0  # pixels
        pinch_score = max(0, (max_pinch_dist - pinch_dist) / max_pinch_dist)
        
        return pinch_score
    
    def _calculate_fist_score(self, landmarks: List[Dict[str, float]]) -> float:
        """Calculate how much the hand looks like a fist."""
        # All fingertips should be close to palm
        palm_center = self._calculate_palm_center(landmarks)
        finger_tips = [4, 8, 12, 16, 20]
        
        tip_distances = []
        for tip_id in finger_tips:
            tip = landmarks[tip_id]
            dist = euclidean([tip['x'], tip['y']], 
                           [palm_center['x'], palm_center['y']])
            tip_distances.append(dist)
        
        # Fist: all tips close to palm
        avg_distance = np.mean(tip_distances)
        max_fist_dist = 60.0  # pixels
        fist_score = max(0, (max_fist_dist - avg_distance) / max_fist_dist)
        
        return fist_score
    
    def _calculate_shape_descriptors(self, landmarks: List[Dict[str, float]]) -> List[float]:
        """Calculate geometric shape descriptors."""
        shape_features = []
        
        # Convex hull properties
        x_coords = [lm['x'] for lm in landmarks]
        y_coords = [lm['y'] for lm in landmarks]
        
        # Bounding box features
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        aspect_ratio = width / (height + 1e-8)
        area = width * height
        
        shape_features.extend([width, height, aspect_ratio, area])
        
        # Centroid features
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        
        # Distance from centroid to extremes
        distances_from_centroid = [
            euclidean([x, y], [centroid_x, centroid_y]) 
            for x, y in zip(x_coords, y_coords)
        ]
        
        shape_features.extend([
            np.mean(distances_from_centroid),
            np.std(distances_from_centroid),
            np.max(distances_from_centroid),
            np.min(distances_from_centroid)
        ])
        
        return shape_features
    
    def _calculate_statistical_features(self, landmarks: List[Dict[str, float]]) -> List[float]:
        """Calculate statistical features of landmark coordinates."""
        x_coords = [lm['x'] for lm in landmarks]
        y_coords = [lm['y'] for lm in landmarks]
        z_coords = [lm['z'] for lm in landmarks]
        
        statistical_features = []
        
        for coords in [x_coords, y_coords, z_coords]:
            statistical_features.extend([
                np.mean(coords),
                np.std(coords),
                np.var(coords),
                stats.skew(coords),
                stats.kurtosis(coords),
                np.percentile(coords, 25),
                np.percentile(coords, 75),
                np.ptp(coords)  # Peak-to-peak (range)
            ])
        
        return statistical_features
    
    def _calculate_topological_features(self, landmarks: List[Dict[str, float]]) -> List[float]:
        """Calculate topological features based on landmark relationships."""
        topological_features = []
        
        # Finger ordering (are fingers in expected order?)
        finger_tips = [4, 8, 12, 16, 20]
        tip_x_coords = [landmarks[tip]['x'] for tip in finger_tips]
        
        # Expected order: thumb, index, middle, ring, pinky (left to right or right to left)
        sorted_indices = np.argsort(tip_x_coords)
        order_consistency = self._calculate_order_consistency(sorted_indices)
        topological_features.append(order_consistency)
        
        # Connectivity patterns (are landmarks forming expected hand structure?)
        connectivity_score = self._calculate_connectivity_score(landmarks)
        topological_features.append(connectivity_score)
        
        return topological_features
    
    def _calculate_order_consistency(self, sorted_indices: np.ndarray) -> float:
        """Calculate how consistent finger ordering is."""
        expected_order = np.array([0, 1, 2, 3, 4])
        expected_reverse = np.array([4, 3, 2, 1, 0])
        
        # Check both forward and reverse order
        consistency_forward = np.mean(sorted_indices == expected_order)
        consistency_reverse = np.mean(sorted_indices == expected_reverse)
        
        return max(consistency_forward, consistency_reverse)
    
    def _calculate_connectivity_score(self, landmarks: List[Dict[str, float]]) -> float:
        """Calculate how well landmarks form expected hand structure."""
        # Simple measure: fingers should diverge from palm
        palm_center = self._calculate_palm_center(landmarks)
        finger_tips = [4, 8, 12, 16, 20]
        
        # Calculate angles between adjacent finger-to-palm vectors
        angles = []
        for i in range(len(finger_tips) - 1):
            tip1 = landmarks[finger_tips[i]]
            tip2 = landmarks[finger_tips[i + 1]]
            
            v1 = np.array([tip1['x'] - palm_center['x'], tip1['y'] - palm_center['y']])
            v2 = np.array([tip2['x'] - palm_center['x'], tip2['y'] - palm_center['y']])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angles.append(angle)
        
        # Good structure: consistent angles between fingers
        angle_std = np.std(angles)
        connectivity_score = 1.0 / (1.0 + angle_std)  # Lower std = higher score
        
        return connectivity_score
    
    def _calculate_angle_between_points(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """Calculate angle between three points."""
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y'], p1['z'] - p2['z']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y'], p3['z'] - p2['z']])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def load_enhanced_training_data(self, data_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load enhanced training data with quality filtering."""
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        features = []
        labels = []
        
        gesture_name_to_id = {name: id for id, name in self.gesture_labels.items()}
        
        for gesture_name, samples in data["data"].items():
            if gesture_name not in gesture_name_to_id:
                logger.warning(f"Unknown gesture in data: {gesture_name}")
                continue
            
            gesture_id = gesture_name_to_id[gesture_name]
            
            for sample in samples:
                try:
                    landmarks = sample["landmarks"]
                    
                    # Quality filtering
                    quality_score = sample.get("quality_score", 0.5)
                    if quality_score < 0.7:  # Skip low quality samples
                        continue
                    
                    feature_vector = self.extract_enhanced_features(landmarks)
                    features.append(feature_vector)
                    labels.append(gesture_id)
                    
                except Exception as e:
                    logger.warning(f"Error processing sample for {gesture_name}: {e}")
                    continue
        
        if len(features) == 0:
            logger.warning("No valid samples found.")
            return np.array([]), np.array([])
        
        logger.info(f"Loaded {len(features)} high-quality samples from {len(set(labels))} gesture classes")
        return np.array(features), np.array(labels)
    
    def _find_latest_enhanced_dataset(self) -> str:
        """Find the latest enhanced dataset file."""
        data_dir = "data/gesture_data"
        if not os.path.exists(data_dir):
            return "data/gesture_data/enhanced_gesture_dataset.json"  # Default fallback
        
        # Look for enhanced dataset files
        dataset_files = [f for f in os.listdir(data_dir) 
                        if f.startswith('enhanced_gesture_dataset_') and f.endswith('.json')]
        
        if not dataset_files:
            # Check for the default name
            default_file = "data/gesture_data/enhanced_gesture_dataset.json"
            return default_file
        
        # Sort by timestamp (newest first)
        dataset_files.sort(reverse=True)
        latest_file = os.path.join(data_dir, dataset_files[0])
        
        return latest_file

    def train_enhanced_model(self, data_file: str = None, 
                           use_advanced_preprocessing: bool = True,
                           use_feature_selection: bool = True,
                           use_pca: bool = True):
        """Train the enhanced model with all optimizations."""
        if data_file is None:
            data_file = self._find_latest_enhanced_dataset()
        
        if not os.path.exists(data_file):
            logger.error(f"Training data file not found: {data_file}")
            return False
        
        # Load data
        logger.info("Loading enhanced training data...")
        features, labels = self.load_enhanced_training_data(data_file)
        
        if len(features) == 0:
            logger.error("No training data available")
            return False
        
        # Data preprocessing
        logger.info("Applying advanced preprocessing...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        if use_feature_selection:
            logger.info("Selecting best features...")
            self.feature_selector = SelectKBest(f_classif, k=min(200, X_train_scaled.shape[1] // 2))
            X_train_scaled = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_scaled = self.feature_selector.transform(X_test_scaled)
        
        # PCA for dimensionality reduction
        if use_pca and X_train_scaled.shape[1] > 50:
            logger.info("Applying PCA...")
            self.pca = PCA(n_components=0.95)  # Retain 95% of variance
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            X_test_scaled = self.pca.transform(X_test_scaled)
        
        # Train ensemble model
        logger.info("Training enhanced ensemble model...")
        start_time = time.time()
        
        if self.use_ensemble:
            self.ensemble_model.fit(X_train_scaled, y_train)
            primary_model = self.ensemble_model
        else:
            # Train just the best single model
            self.base_models['rf'].fit(X_train_scaled, y_train)
            primary_model = self.base_models['rf']
        
        training_time = time.time() - start_time
        
        # Evaluate
        y_pred = primary_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(primary_model, X_train_scaled, y_train, cv=5)
        
        # Store training statistics
        self.training_stats = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time,
            'num_features': X_train_scaled.shape[1],
            'num_samples': len(X_train)
        }
        
        logger.info(f"Enhanced model trained successfully!")
        logger.info(f"Test accuracy: {accuracy:.3f}")
        logger.info(f"Cross-validation: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        logger.info(f"Training time: {training_time:.2f}s")
        logger.info(f"Features used: {X_train_scaled.shape[1]}")
        
        # Feature importance (if using Random Forest)
        if hasattr(primary_model, 'feature_importances_'):
            self.feature_importance = primary_model.feature_importances_
        
        self.is_trained = True
        return True
    
    def predict_gesture(self, landmarks: List[Dict[str, float]]) -> Dict[str, Any]:
        """Predict gesture with enhanced accuracy and temporal smoothing."""
        if not self.is_trained:
            return {"gesture": "unknown", "confidence": 0.0, "error": "Model not trained"}
        
        try:
            # Extract enhanced features
            features = self.extract_enhanced_features(landmarks)
            
            # Apply preprocessing pipeline
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            if self.feature_selector is not None:
                features_scaled = self.feature_selector.transform(features_scaled)
            
            if self.pca is not None:
                features_scaled = self.pca.transform(features_scaled)
            
            # Get prediction from ensemble
            if self.use_ensemble:
                prediction = self.ensemble_model.predict(features_scaled)[0]
                probabilities = self.ensemble_model.predict_proba(features_scaled)[0]
            else:
                prediction = self.base_models['rf'].predict(features_scaled)[0]
                probabilities = self.base_models['rf'].predict_proba(features_scaled)[0]
            
            confidence = np.max(probabilities)
            
            # Temporal smoothing
            if self.use_temporal:
                self.temporal_buffer.append({
                    'prediction': prediction,
                    'confidence': confidence,
                    'probabilities': probabilities
                })
                
                # Smooth predictions
                prediction, confidence, probabilities = self._smooth_temporal_predictions()
            
            gesture_name = self.gesture_labels.get(prediction, "unknown")
            
            return {
                "gesture": gesture_name,
                "confidence": float(confidence),
                "type": "enhanced",
                "probabilities": {self.gesture_labels[i]: float(prob) 
                                for i, prob in enumerate(probabilities)},
                "temporal_buffer_size": len(self.temporal_buffer)
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced gesture prediction: {str(e)}")
            return {"gesture": "error", "confidence": 0.0, "error": str(e)}
    
    def _smooth_temporal_predictions(self) -> Tuple[int, float, np.ndarray]:
        """Apply temporal smoothing to predictions."""
        if len(self.temporal_buffer) < 2:
            last_pred = self.temporal_buffer[-1]
            return last_pred['prediction'], last_pred['confidence'], last_pred['probabilities']
        
        # Weighted average based on confidence
        predictions = [pred['prediction'] for pred in self.temporal_buffer]
        confidences = [pred['confidence'] for pred in self.temporal_buffer]
        prob_arrays = [pred['probabilities'] for pred in self.temporal_buffer]
        
        # More recent predictions get higher weight
        weights = np.linspace(0.5, 1.0, len(self.temporal_buffer))
        weights = weights * np.array(confidences)  # Also weight by confidence
        weights = weights / np.sum(weights)
        
        # Weighted average of probabilities
        avg_probabilities = np.average(prob_arrays, weights=weights, axis=0)
        
        # Final prediction
        final_prediction = np.argmax(avg_probabilities)
        final_confidence = np.max(avg_probabilities)
        
        return final_prediction, final_confidence, avg_probabilities
    
    def save_enhanced_model(self, filepath: str):
        """Save the enhanced model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'ensemble_model': self.ensemble_model,
            'base_models': self.base_models,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'pca': self.pca,
            'gesture_labels': self.gesture_labels,
            'use_ensemble': self.use_ensemble,
            'use_temporal': self.use_temporal,
            'temporal_window': self.temporal_window,
            'training_stats': self.training_stats,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Enhanced model saved to {filepath}")
    
    def load_enhanced_model(self, filepath: str):
        """Load an enhanced model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.ensemble_model = model_data.get('ensemble_model')
        self.base_models = model_data.get('base_models', {})
        self.scaler = model_data['scaler']
        self.feature_selector = model_data.get('feature_selector')
        self.pca = model_data.get('pca')
        self.gesture_labels = model_data['gesture_labels']
        self.use_ensemble = model_data.get('use_ensemble', True)
        self.use_temporal = model_data.get('use_temporal', True)
        self.temporal_window = model_data.get('temporal_window', 10)
        self.training_stats = model_data.get('training_stats', {})
        self.feature_importance = model_data.get('feature_importance', {})
        
        self.is_trained = True
        logger.info(f"Enhanced model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the model."""
        return {
            "model_type": "enhanced_ensemble",
            "use_ensemble": self.use_ensemble,
            "use_temporal": self.use_temporal,
            "is_trained": self.is_trained,
            "gesture_count": len(self.gesture_labels),
            "supported_gestures": list(self.gesture_labels.values()),
            "training_stats": self.training_stats,
            "temporal_window": self.temporal_window,
            "base_models": list(self.base_models.keys()) if self.base_models else []
        } 