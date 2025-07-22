"""
Camera Panel Widget
==================

Right panel widget for camera feed and gesture detection.
Integrates with existing gesture detection system from core modules.
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QGroupBox, QCheckBox,
                             QSlider, QSpinBox, QTextEdit)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont

import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

# Also add the hand_tracking directory for direct imports
hand_tracking_path = src_path / "core" / "hand_tracking"
sys.path.insert(0, str(hand_tracking_path))

# Try multiple import strategies for robustness
try:
    # First try: direct import from hand_tracking directory
    from core.gesture_detector import GestureDetector
    from core.gesture_classifier_adapter import GestureClassifier
    print("✅ Using direct imports from hand_tracking directory")
except ImportError:
    try:
        # Second try: full path import
        from core.hand_tracking.core.gesture_detector import GestureDetector
        from core.hand_tracking.core.gesture_classifier_adapter import GestureClassifier
        print("✅ Using full path imports")
    except ImportError:
        # Fallback: basic functionality without enhanced features
        print("⚠️  Enhanced gesture classifier not available - using basic mode")
        GestureDetector = None
        GestureClassifier = None


class CameraWorker(QThread):
    """Worker thread for camera processing to avoid UI blocking."""
    
    frame_ready = pyqtSignal(np.ndarray, dict)  # Frame and detection results
    error_occurred = pyqtSignal(str)  # Error message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera = None
        self.gesture_detector = None
        self.gesture_classifier = None
        self.running = False
        
        # Settings
        self.show_landmarks = True
        self.show_connections = True
        self.enable_classification = False
        self.enable_gesture_detection = False  # Start with gesture detection disabled
    
    def initialize_camera(self, camera_index=0):
        """Initialize camera and gesture detection."""
        try:
            # Initialize camera first
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                raise Exception("Could not open camera")
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Initialize gesture detector only if enabled
            self.gesture_detector = None
            if self.enable_gesture_detection:
                try:
                    self.gesture_detector = GestureDetector()
                    config = {
                        "static_image_mode": False,
                        "max_num_hands": 2,
                        "min_detection_confidence": 0.7,
                        "min_tracking_confidence": 0.5
                    }
                    
                    if not self.gesture_detector.initialize(config):
                        self.error_occurred.emit("Gesture detector initialization failed - continuing with camera only")
                        self.gesture_detector = None
                        
                except Exception as e:
                    self.error_occurred.emit(f"Gesture detector error: {str(e)} - continuing with camera only")
                    self.gesture_detector = None
            else:
                self.error_occurred.emit("Gesture detection disabled - camera only mode")
            
            # Don't initialize gesture classifier by default to avoid crashes
            self.gesture_classifier = None
            if self.enable_classification and self.gesture_detector:
                try:
                    self.gesture_classifier = GestureClassifier(
                        model_type='knn', 
                        use_feature_selection=True, 
                        n_features=50
                    )
                    self.gesture_classifier.train()
                except Exception as e:
                    self.error_occurred.emit(f"Gesture classifier failed: {str(e)}")
                    self.gesture_classifier = None
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Camera initialization failed: {str(e)}")
            return False
    
    def run(self):
        """Main camera processing loop."""
        self.running = True
        
        while self.running:
            if self.camera is None:
                break
                
            ret, frame = self.camera.read()
            if not ret:
                self.error_occurred.emit("Failed to read frame from camera")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            results = self.process_frame(frame)
            
            # Emit frame and results
            self.frame_ready.emit(frame, results)
            
            # Small delay to control frame rate
            self.msleep(33)  # ~30 FPS
    
    def process_frame(self, frame):
        """Process frame for hand detection and gesture recognition."""
        results = {"hands_detected": False, "hands": []}
        
        try:
            if self.gesture_detector:
                # Detect hands
                detection_results = self.gesture_detector.detect_hands(frame)
                
                if detection_results.get("hands_detected"):
                    results = detection_results.copy()
                    
                    # Add gesture classification if enabled
                    if self.gesture_classifier and results.get("hands"):
                        for hand_data in results["hands"]:
                            landmarks = hand_data["landmarks"]
                            gesture_result = self.gesture_classifier.predict_gesture(landmarks)
                            hand_data["gesture"] = gesture_result
                    
                    # Draw annotations
                    if self.show_connections:
                        frame = self.gesture_detector.draw_connections(frame, results)
                    
                    if self.show_landmarks:
                        frame = self.gesture_detector.draw_landmarks(frame, results)
                        
        except Exception as e:
            self.error_occurred.emit(f"Frame processing error: {str(e)}")
        
        return results
    
    def stop(self):
        """Stop the camera worker."""
        self.running = False
        if self.camera:
            self.camera.release()
            self.camera = None


class CameraPanel(QFrame):
    """Camera panel widget for gesture detection and video display."""
    
    # Signals
    gesture_detected = pyqtSignal(dict)  # Gesture data signal
    camera_status_changed = pyqtSignal(str)  # Camera status signal
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setMinimumWidth(500)
        
        # Camera worker
        self.camera_worker = CameraWorker()
        self.camera_worker.frame_ready.connect(self.update_frame)
        self.camera_worker.error_occurred.connect(self.handle_error)
        
        # Camera state
        self.camera_active = False
        self.current_frame = None
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the camera panel user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("🖐️ Hand Gesture Detection")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Camera controls
        self.setup_camera_controls(layout)
        
        # Video display
        self.setup_video_display(layout)
        
        # Gesture information
        self.setup_gesture_info(layout)
        
        # Detection settings
        self.setup_detection_settings(layout)
    
    def setup_camera_controls(self, layout):
        """Setup camera control buttons."""
        controls_group = QGroupBox("Camera Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        # Start camera button
        self.start_button = QPushButton("📹 Start Camera")
        self.start_button.clicked.connect(self.start_camera)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        controls_layout.addWidget(self.start_button)
        
        # Stop camera button
        self.stop_button = QPushButton("⏹️ Stop Camera")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                border: none;
                color: white;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
        """)
        controls_layout.addWidget(self.stop_button)
        
        layout.addWidget(controls_group)
    
    def setup_video_display(self, layout):
        """Setup video display area."""
        video_group = QGroupBox("Live Video Feed")
        video_layout = QVBoxLayout(video_group)
        
        # Video label
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border: 2px solid #555555;
                border-radius: 5px;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Camera not active")
        video_layout.addWidget(self.video_label)
        
        # Statistics
        stats_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: 0")
        self.detection_rate_label = QLabel("Detection Rate: 0%")
        self.frame_count_label = QLabel("Frames: 0")
        
        for label in [self.fps_label, self.detection_rate_label, self.frame_count_label]:
            label.setStyleSheet("color: #cccccc; padding: 5px;")
            stats_layout.addWidget(label)
        
        video_layout.addLayout(stats_layout)
        layout.addWidget(video_group)
    
    def setup_gesture_info(self, layout):
        """Setup gesture information display."""
        gesture_group = QGroupBox("🎯 Current Gesture")
        gesture_layout = QVBoxLayout(gesture_group)
        
        # Current gesture display
        self.current_gesture = QLabel("No hands detected")
        self.current_gesture.setStyleSheet("""
            QLabel {
                background-color: #404040;
                padding: 15px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        gesture_layout.addWidget(self.current_gesture)
        
        layout.addWidget(gesture_group)
    
    def setup_detection_settings(self, layout):
        """Setup detection settings."""
        settings_group = QGroupBox("Detection Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Show landmarks checkbox
        self.landmarks_checkbox = QCheckBox("Show Hand Landmarks")
        self.landmarks_checkbox.setChecked(True)
        self.landmarks_checkbox.stateChanged.connect(self.update_settings)
        settings_layout.addWidget(self.landmarks_checkbox)
        
        # Show connections checkbox
        self.connections_checkbox = QCheckBox("Show Hand Connections")
        self.connections_checkbox.setChecked(True)
        self.connections_checkbox.stateChanged.connect(self.update_settings)
        settings_layout.addWidget(self.connections_checkbox)
        
        # Enable gesture detection checkbox
        self.detection_checkbox = QCheckBox("Enable Gesture Detection")
        self.detection_checkbox.setChecked(False)
        self.detection_checkbox.stateChanged.connect(self.update_settings)
        settings_layout.addWidget(self.detection_checkbox)
        
        # Enable gesture classification checkbox
        self.classification_checkbox = QCheckBox("Enable Gesture Classification")
        self.classification_checkbox.setChecked(False)
        self.classification_checkbox.stateChanged.connect(self.update_settings)
        settings_layout.addWidget(self.classification_checkbox)
        
        layout.addWidget(settings_group)
    
    def start_camera(self):
        """Start camera and gesture detection."""
        try:
            if self.camera_worker.initialize_camera():
                self.camera_worker.start()
                self.camera_active = True
                
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                
                self.camera_status_changed.emit("✅ Camera active")
                
                # Reset statistics
                self.frame_count = 0
                self.detection_count = 0
            else:
                self.camera_status_changed.emit("❌ Camera failed to start")
                
        except Exception as e:
            self.handle_error(f"Start camera error: {str(e)}")
    
    def stop_camera(self):
        """Stop camera and gesture detection."""
        try:
            self.camera_worker.stop()
            self.camera_worker.wait()  # Wait for thread to finish
            
            self.camera_active = False
            
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            
            self.video_label.setText("Camera not active")
            self.current_gesture.setText("No hands detected")
            
            self.camera_status_changed.emit("📷 Camera inactive")
            
        except Exception as e:
            self.handle_error(f"Stop camera error: {str(e)}")
    
    @pyqtSlot(np.ndarray, dict)
    def update_frame(self, frame, results):
        """Update video display with new frame and detection results."""
        try:
            # Update statistics
            self.frame_count += 1
            if results.get("hands_detected"):
                self.detection_count += 1
            
            # Convert frame to QPixmap
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale image to fit label
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
            
            # Update gesture information
            self.update_gesture_display(results)
            
            # Update statistics display
            self.update_statistics()
            
        except Exception as e:
            self.handle_error(f"Frame update error: {str(e)}")
    
    def update_gesture_display(self, results):
        """Update gesture display with detection results."""
        if not results.get("hands_detected"):
            self.current_gesture.setText("No hands detected")
            return
        
        hands = results.get("hands", [])
        if not hands:
            return
        
        # Display information for the first detected hand
        hand = hands[0]
        handedness = hand.get("handedness", {}).get("label", "Unknown")
        confidence = hand.get("handedness", {}).get("score", 0.0)
        
        # Check if gesture classification is available
        if "gesture" in hand:
            gesture_info = hand["gesture"]
            gesture_name = gesture_info.get("gesture", "unknown")
            gesture_confidence = gesture_info.get("confidence", 0.0)
            
            display_text = f"🖐️ {handedness} hand: {gesture_name}\nConfidence: {gesture_confidence:.2f}"
            
            # Emit gesture signal for data panel
            self.gesture_detected.emit({
                "gesture": gesture_name,
                "confidence": gesture_confidence,
                "handedness": handedness,
                "hand_confidence": confidence
            })
        else:
            display_text = f"🖐️ {handedness} hand detected\nConfidence: {confidence:.2f}"
        
        self.current_gesture.setText(display_text)
    
    def update_statistics(self):
        """Update statistics display."""
        if self.frame_count > 0:
            detection_rate = (self.detection_count / self.frame_count) * 100
            self.detection_rate_label.setText(f"Detection Rate: {detection_rate:.1f}%")
        
        self.frame_count_label.setText(f"Frames: {self.frame_count}")
        
        # Simple FPS calculation (approximate)
        fps = 30  # Assuming ~30 FPS
        self.fps_label.setText(f"FPS: {fps}")
    
    def update_settings(self):
        """Update detection settings."""
        if hasattr(self.camera_worker, 'show_landmarks'):
            self.camera_worker.show_landmarks = self.landmarks_checkbox.isChecked()
            self.camera_worker.show_connections = self.connections_checkbox.isChecked()
            self.camera_worker.enable_gesture_detection = self.detection_checkbox.isChecked()
            self.camera_worker.enable_classification = self.classification_checkbox.isChecked()
    
    @pyqtSlot(str)
    def handle_error(self, error_message):
        """Handle error messages from camera worker."""
        print(f"Camera error: {error_message}")
        self.camera_status_changed.emit(f"❌ {error_message}")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.camera_active:
            self.stop_camera() 