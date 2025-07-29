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
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Compact title with camera controls in same line
        header_layout = QHBoxLayout()
        
        title = QLabel("📹 Webcam")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: #ffffff;")
        header_layout.addWidget(title)
        
        # Compact start/stop buttons
        self.start_button = QPushButton("▶️")
        self.start_button.clicked.connect(self.start_camera)
        self.start_button.setFixedSize(32, 32)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                border-radius: 16px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        header_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("⏹️")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)
        self.stop_button.setFixedSize(32, 32)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                border: none;
                color: white;
                border-radius: 16px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
        """)
        header_layout.addWidget(self.stop_button)
        
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Bigger video display - this is the main focus
        self.setup_video_display(layout)
        
        # Minimal gesture info (just current gesture, no extra sections)
        self.setup_minimal_gesture_info(layout)

    def setup_video_display(self, layout):
        """Setup large, prominent video display area."""
        # Video display container
        video_container = QFrame()
        video_container.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                padding: 4px;
            }
        """)
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(4, 4, 4, 4)
        
        # Main video label - bigger and better
        self.video_label = QLabel()
        self.video_label.setMinimumSize(480, 360)  # Better aspect ratio, bigger size
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border: 1px solid #333333;
                border-radius: 4px;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("👤 Camera Off\nClick ▶️ to start")
        self.video_label.setScaledContents(True)  # Scale content to fit
        video_layout.addWidget(self.video_label)
        
        # Minimal stats in one line
        stats_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: --")
        self.gesture_status_label = QLabel("Gesture: --")
        
        for label in [self.fps_label, self.gesture_status_label]:
            label.setStyleSheet("""
                color: #888888; 
                font-size: 11px;
                padding: 2px;
            """)
            stats_layout.addWidget(label)
        
        stats_layout.addStretch()
        video_layout.addLayout(stats_layout)
        
        layout.addWidget(video_container)
    
    def setup_minimal_gesture_info(self, layout):
        """Setup minimal gesture information display."""
        # Just current gesture in a compact format
        gesture_container = QFrame()
        gesture_container.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        gesture_layout = QHBoxLayout(gesture_container)
        gesture_layout.setContentsMargins(8, 6, 8, 6)
        
        gesture_icon = QLabel("🖐️")
        gesture_icon.setStyleSheet("font-size: 16px;")
        gesture_layout.addWidget(gesture_icon)
        
        self.current_gesture = QLabel("Ready for gestures")
        self.current_gesture.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        gesture_layout.addWidget(self.current_gesture)
        
        gesture_layout.addStretch()
        layout.addWidget(gesture_container)

    # Remove the old setup methods and replace with streamlined versions
    def setup_camera_controls(self, layout):
        """Legacy method - now handled in main setup_ui"""
        pass
    
    def setup_gesture_info(self, layout):
        """Legacy method - now handled in setup_minimal_gesture_info"""
        pass
    
    def start_camera(self):
        """Start camera and gesture detection with detection/classification enabled by default."""
        try:
            # Enable detection and classification by default
            self.camera_worker.enable_gesture_detection = True
            self.camera_worker.enable_classification = True
            
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
            
            self.video_label.setText("👤 Camera Off\nClick ▶️ to start")
            self.current_gesture.setText("Ready for gestures")
            
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
            # Emit signal to indicate no gesture detected
            self.gesture_detected.emit({
                "gesture": "none",
                "confidence": 0.0,
                "handedness": "none",
                "hand_confidence": 0.0
            })
            return
        
        hands = results.get("hands", [])
        if not hands:
            # Emit signal to indicate no gesture detected
            self.gesture_detected.emit({
                "gesture": "none",
                "confidence": 0.0,
                "handedness": "none",
                "hand_confidence": 0.0
            })
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
            
            # Emit gesture signal
            self.gesture_detected.emit({
                "gesture": gesture_name,
                "confidence": gesture_confidence,
                "handedness": handedness,
                "hand_confidence": confidence
            })
        else:
            display_text = f"🖐️ {handedness} hand detected\nConfidence: {confidence:.2f}"
            # Emit signal to indicate hand detected but no gesture classified
            self.gesture_detected.emit({
                "gesture": "none",
                "confidence": 0.0,
                "handedness": handedness,
                "hand_confidence": confidence
            })
        
        self.current_gesture.setText(display_text)
    
    def update_statistics(self):
        """Update simplified statistics display."""
        # Simple FPS calculation (approximate)
        fps = 30  # Assuming ~30 FPS
        self.fps_label.setText(f"FPS: {fps}")
        
        # Update gesture status
        if self.frame_count > 0:
            detection_rate = (self.detection_count / self.frame_count) * 100
            self.gesture_status_label.setText(f"Detection: {detection_rate:.0f}%")
    
    def update_settings(self):
        """No-op: Detection settings UI removed."""
        pass
    
    @pyqtSlot(str)
    def handle_error(self, error_message):
        """Handle error messages from camera worker."""
        print(f"Camera error: {error_message}")
        self.camera_status_changed.emit(f"❌ {error_message}")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.camera_active:
            self.stop_camera() 