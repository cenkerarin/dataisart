"""
Hand Tracking Test Dashboard
===========================

Streamlit dashboard for testing hand tracking and gesture detection using
the existing hand tracking modules with real camera feed integration.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import sys
import os
from pathlib import Path

# Add the hand_tracking directory to Python path for imports
hand_tracking_path = Path(__file__).parent.parent.parent / "core" / "hand_tracking"
sys.path.insert(0, str(hand_tracking_path))

# Import existing hand tracking modules
from core.gesture_detector import GestureDetector
from core.gesture_classifier import GestureClassifier

class HandTrackingTestDashboard:
    """Test dashboard for hand tracking and gesture detection."""
    
    def __init__(self):
        """Initialize the test dashboard."""
        self.gesture_detector = None
        self.gesture_classifier = None
        self.camera = None
        self.is_camera_active = False
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.gesture_history = []
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Display options
        self.show_landmarks = True
        self.show_connections = True
        self.show_bounding_box = True
        
    def initialize_components(self):
        """Initialize hand tracking and gesture classification components with error handling."""
        
        # Initialize gesture detector
        if 'gesture_detector' not in st.session_state:
            try:
                st.session_state.gesture_detector = GestureDetector()
                
                # Initialize with configuration from demo
                config = {
                    "static_image_mode": False,
                    "max_num_hands": 2,
                    "min_detection_confidence": 0.7,
                    "min_tracking_confidence": 0.5
                }
                
                if st.session_state.gesture_detector.initialize(config):
                    st.success("✅ Gesture detector initialized successfully")
                else:
                    st.error("❌ Failed to initialize gesture detector")
                    st.session_state.gesture_detector = None
                    
            except Exception as e:
                st.error(f"❌ Gesture detector initialization error: {str(e)}")
                st.session_state.gesture_detector = None
        
        # Initialize gesture classifier (optional - make it more robust)
        if 'gesture_classifier' not in st.session_state:
            classifier_enabled = st.sidebar.checkbox(
                "🤖 Enable Gesture Classification", 
                value=False,
                help="Enable AI-powered gesture recognition (may impact performance)"
            )
            
            if classifier_enabled:
                try:
                    with st.spinner("Training gesture classifier..."):
                        st.session_state.gesture_classifier = GestureClassifier(
                            model_type='knn', 
                            use_feature_selection=True, 
                            n_features=50
                        )
                        # Train the classifier
                        st.session_state.gesture_classifier.train()
                        st.success("✅ Gesture classifier initialized and trained")
                except Exception as e:
                    st.warning(f"⚠️ Gesture classifier initialization failed: {str(e)}")
                    st.warning("📋 Continuing with hand tracking only (no gesture classification)")
                    st.session_state.gesture_classifier = None
            else:
                st.session_state.gesture_classifier = None
                if classifier_enabled is False:  # Only show this message if explicitly disabled
                    st.info("ℹ️ Gesture classification disabled - only hand tracking active")
    
    def start_camera(self):
        """Start camera capture using OpenCV with robust error handling."""
        try:
            # Release any existing camera first
            if 'camera' in st.session_state and st.session_state.camera is not None:
                try:
                    st.session_state.camera.release()
                except:
                    pass
                st.session_state.camera = None
            
            # Try to initialize camera
            with st.spinner("Initializing camera..."):
                camera = cv2.VideoCapture(0)
                
                if not camera.isOpened():
                    # Try alternative camera indices
                    for i in range(1, 4):
                        camera.release()
                        camera = cv2.VideoCapture(i)
                        if camera.isOpened():
                            st.info(f"📹 Using camera index {i}")
                            break
                    else:
                        st.error("❌ Could not open any camera. Please check camera permissions and connections.")
                        return False
                
                # Test camera by reading one frame
                ret, test_frame = camera.read()
                if not ret or test_frame is None:
                    st.error("❌ Camera opened but cannot read frames")
                    camera.release()
                    return False
                
                # Set camera properties
                try:
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    camera.set(cv2.CAP_PROP_FPS, 30)  # Set target FPS
                    
                    # Verify settings
                    actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                    actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    st.info(f"📹 Camera resolution: {int(actual_width)}x{int(actual_height)}")
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not set camera properties: {str(e)}")
                
                st.session_state.camera = camera
                
            # Reset statistics
            st.session_state.camera_active = True
            st.session_state.frame_count = 0
            st.session_state.detection_count = 0
            st.session_state.gesture_history = []
            st.session_state.current_fps = 0.0
            
            # Clear any previous error states
            if 'last_fps_update' in st.session_state:
                del st.session_state.last_fps_update
            if 'fps_frame_count' in st.session_state:
                del st.session_state.fps_frame_count
            
            st.success("✅ Camera started successfully!")
            return True
            
        except Exception as e:
            st.error(f"❌ Error starting camera: {str(e)}")
            st.error("💡 Try: Check camera permissions, close other camera apps, or restart the dashboard")
            return False
    
    def stop_camera(self):
        """Stop camera capture safely."""
        try:
            if 'camera' in st.session_state and st.session_state.camera is not None:
                st.session_state.camera.release()
                st.session_state.camera = None
            
            st.session_state.camera_active = False
            st.session_state.live_feed_enabled = False
            
            # Clear any refresh timers
            if 'last_refresh_time' in st.session_state:
                del st.session_state.last_refresh_time
            if 'refresh_counter' in st.session_state:
                del st.session_state.refresh_counter
                
            st.success("📷 Camera stopped successfully")
            
        except Exception as e:
            st.warning(f"⚠️ Error stopping camera: {str(e)}")
            # Force cleanup even if there's an error
            st.session_state.camera_active = False
            st.session_state.camera = None
    
    def capture_and_process_frame(self):
        """Capture and process a single frame using existing detection code with error handling."""
        try:
            if not st.session_state.get('camera_active', False):
                st.warning("🚫 Camera not active in session state")
                return None, None
                
            camera = st.session_state.get('camera')
            if camera is None:
                st.error("❌ Camera object is None")
                return None, None
                
            if not camera.isOpened():
                st.error("❌ Camera is not opened")
                return None, None
                
            ret, frame = camera.read()
            if not ret:
                st.warning("⚠️ Could not read frame from camera (ret=False)")
                return None, None
                
            if frame is None:
                st.warning("⚠️ Frame is None")
                return None, None
            
            # Flip frame horizontally for mirror effect (like demo)
            frame = cv2.flip(frame, 1)
            
            # Detect hands using existing detector with error handling
            detector = st.session_state.gesture_detector
            if not detector or not detector.is_initialized:
                st.error("❌ Gesture detector not initialized")
                return frame, None
            
            try:
                results = detector.detect_hands(frame)
            except Exception as e:
                st.error(f"❌ Hand detection error: {str(e)}")
                return frame, None
            
            # Process with gesture classifier if available (with error handling)
            if (st.session_state.get('gesture_classifier') and 
                results and results.get("hands_detected")):
                try:
                    for hand_data in results["hands"]:
                        landmarks = hand_data["landmarks"]
                        gesture_result = st.session_state.gesture_classifier.predict_gesture(landmarks)
                        hand_data["gesture"] = gesture_result
                        
                        # Add to history safely
                        history_entry = {
                            "timestamp": time.time(),
                            "gesture": gesture_result.get("gesture", "unknown"),
                            "confidence": float(gesture_result.get("confidence", 0.0)),
                            "handedness": hand_data["handedness"]["label"]
                        }
                        
                        if 'gesture_history' not in st.session_state:
                            st.session_state.gesture_history = []
                        st.session_state.gesture_history.append(history_entry)
                        
                        # Keep only last 50 entries to prevent memory issues
                        if len(st.session_state.gesture_history) > 50:
                            st.session_state.gesture_history = st.session_state.gesture_history[-50:]
                            
                except Exception as e:
                    st.warning(f"⚠️ Gesture classification error: {str(e)}")
                    # Continue without gesture classification
            
            # Draw annotations using existing methods with error handling
            display_frame = frame.copy()
            
            try:
                # Use the detector's drawing methods from the demo
                if self.show_connections and results:
                    display_frame = detector.draw_connections(display_frame, results)
                
                if self.show_landmarks and results:
                    display_frame = detector.draw_landmarks(display_frame, results)
            except Exception as e:
                st.warning(f"⚠️ Drawing error: {str(e)}")
                # Use original frame if drawing fails
                display_frame = frame
            
            # Update statistics safely
            if 'frame_count' not in st.session_state:
                st.session_state.frame_count = 0
            if 'detection_count' not in st.session_state:
                st.session_state.detection_count = 0
                
            st.session_state.frame_count += 1
            if results and results.get("hands_detected"):
                st.session_state.detection_count += 1
            
            # Calculate FPS (simplified and more stable)
            current_time = time.time()
            if 'last_fps_update' not in st.session_state:
                st.session_state.last_fps_update = current_time
                st.session_state.fps_frame_count = 0
                st.session_state.current_fps = 0.0
            
            st.session_state.fps_frame_count += 1
            
            # Update FPS every 10 frames instead of 30 for more responsive updates
            if st.session_state.fps_frame_count >= 10:
                elapsed = current_time - st.session_state.last_fps_update
                if elapsed > 0:
                    st.session_state.current_fps = st.session_state.fps_frame_count / elapsed
                st.session_state.last_fps_update = current_time
                st.session_state.fps_frame_count = 0
            
            # Add FPS display
            fps = st.session_state.get('current_fps', 0.0)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return display_frame, results
            
        except Exception as e:
            st.error(f"❌ Frame processing error: {str(e)}")
            return None, None
    
    def render(self):
        """Render the main dashboard."""
        st.title("🖐️ Hand Tracking Test Dashboard")
        st.markdown("*Test and debug hand tracking with real camera feed*")
        
        # Initialize components
        self.initialize_components()
        
        # Sidebar controls
        self.render_sidebar()
        
        # Main content
        col1, col2 = st.columns([3, 2])
        
        with col1:
            self.render_camera_panel()
        
        with col2:
            st.header("📊 Real-Time Statistics")
            # Create fresh placeholders each time for better stability
            stats_placeholder = st.empty()
            history_placeholder = st.empty()
            chart_placeholder = st.empty()
            
            # Update statistics in real-time
            self.update_stats_display(stats_placeholder, history_placeholder, chart_placeholder)
        
        # Status bar
        self.render_status_bar()
        
        # Simple auto-refresh for live camera feed
        camera_active = st.session_state.get('camera_active', False)
        live_feed_enabled = st.session_state.get('live_feed_enabled', False)
        
        if camera_active and live_feed_enabled:
            # Simple 1-second refresh for stable performance
            time.sleep(1.0)
            st.rerun()
    
    def render_sidebar(self):
        """Render sidebar controls."""
        with st.sidebar:
            st.header("🎛️ Camera Controls")
            
            # Camera controls
            camera_active = st.session_state.get('camera_active', False)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📹 Start Camera", disabled=camera_active):
                    if self.start_camera():
                        st.rerun()
            
            with col2:
                if st.button("⏹️ Stop Camera", disabled=not camera_active):
                    self.stop_camera()
                    st.rerun()
            
            # Detection settings
            st.subheader("⚙️ Detection Settings")
            
            max_hands = st.slider("Max Hands", 1, 2, 2)
            min_detection_confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.7, 0.1)
            min_tracking_confidence = st.slider("Tracking Confidence", 0.1, 1.0, 0.5, 0.1)
            
            if st.button("Update Settings"):
                config = {
                    "static_image_mode": False,
                    "max_num_hands": max_hands,
                    "min_detection_confidence": min_detection_confidence,
                    "min_tracking_confidence": min_tracking_confidence
                }
                if st.session_state.gesture_detector.initialize(config):
                    st.success("Settings updated!")
                else:
                    st.error("Failed to update settings")
            
            # Display options
            st.subheader("🎨 Display Options")
            
            self.show_landmarks = st.checkbox("Show Landmarks", value=True)
            self.show_connections = st.checkbox("Show Connections", value=True)
            self.show_bounding_box = st.checkbox("Show Bounding Box", value=True)
            
            # Clear history
            st.subheader("🗑️ Data Management")
            if st.button("Clear Gesture History"):
                st.session_state.gesture_history = []
                st.session_state.frame_count = 0
                st.session_state.detection_count = 0
                st.success("History cleared!")
    
    def render_camera_panel(self):
        """Render the camera feed panel."""
        st.header("📹 Live Camera Feed")
        
        camera_active = st.session_state.get('camera_active', False)
        
        if not camera_active:
            st.info("📷 Camera is inactive. Click 'Start Camera' in the sidebar.")
            
            # Show instructions
            st.subheader("🎯 Testing Instructions:")
            st.markdown("""
            1. **Start Camera**: Click the start button in sidebar
            2. **Position Hands**: Place hands in front of camera
            3. **Test Gestures**: Try different hand positions and gestures
            4. **Monitor Stats**: Check detection accuracy in the right panel
            5. **Adjust Settings**: Tune confidence thresholds as needed
            """)
            
        else:
            # Real-time camera feed
            camera_placeholder = st.empty()
            detection_info_placeholder = st.empty()
            debug_info_placeholder = st.empty()
            
            # Control buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                live_feed = st.checkbox("🔴 Live Feed", value=True, help="Enable continuous video feed")
                st.session_state.live_feed_enabled = live_feed  # Store in session state
            
            with col2:
                fps_limit = st.selectbox("FPS Limit", [5, 10, 15, 30], index=1, help="Frames per second")
            
            with col3:
                show_debug = st.checkbox("🐛 Show Debug Info", value=True)  # Default to True for debugging
            
            # Debug information
            if show_debug:
                with debug_info_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Camera Active", "✅" if st.session_state.get('camera_active', False) else "❌")
                    with col2:
                        st.metric("Live Feed", "✅" if live_feed else "❌")
                    with col3:
                        camera_obj = st.session_state.get('camera')
                        st.metric("Camera Object", "✅" if camera_obj and camera_obj.isOpened() else "❌")
            
            # Real-time video loop
            if live_feed:
                self.run_live_feed(camera_placeholder, detection_info_placeholder, fps_limit, show_debug)
            else:
                # Manual capture mode
                if st.button("📸 Capture Single Frame"):
                    display_frame, results = self.capture_and_process_frame()
                    
                    if display_frame is not None:
                        # Convert BGR to RGB for Streamlit
                        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        camera_placeholder.image(display_frame_rgb, channels="RGB", use_column_width=True)
                        
                        # Show detection results
                        self.show_detection_results(detection_info_placeholder, results, show_debug)
    
    def run_live_feed(self, camera_placeholder, detection_info_placeholder, fps_limit, show_debug):
        """Run continuous live camera feed - simplified for stability."""
        
        try:
            # Process just one frame per refresh cycle for stability
            if not st.session_state.get('camera_active', False):
                detection_info_placeholder.warning("📷 Camera inactive")
                return
                
            display_frame, results = self.capture_and_process_frame()
            
            if display_frame is not None:
                try:
                    # Convert BGR to RGB for Streamlit
                    display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(display_frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Show detection results
                    self.show_detection_results(detection_info_placeholder, results, show_debug)
                    
                except Exception as e:
                    st.warning(f"⚠️ Display error: {str(e)}")
                    
            else:
                detection_info_placeholder.warning("⚠️ No frame received from camera")
                
        except Exception as e:
            st.error(f"❌ Live feed error: {str(e)}")
            detection_info_placeholder.error("Live feed temporarily unavailable")
    
    def show_detection_results(self, placeholder, results, show_debug):
        """Display detection results in real-time."""
        if not results:
            placeholder.warning("👋 No hands detected")
            return
            
        if results.get("hands_detected"):
            hands_count = len(results['hands'])
            
            with placeholder.container():
                st.success(f"✅ {hands_count} hand(s) detected")
                
                if show_debug:
                    for i, hand in enumerate(results["hands"]):
                        with st.expander(f"Hand {i+1} - {hand['handedness']['label']}", expanded=False):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Confidence**: {hand['handedness']['score']:.3f}")
                                bbox = hand['bounding_box']
                                st.write(f"**Bounding Box**: {bbox['width']}×{bbox['height']}")
                            
                            with col2:
                                if 'gesture' in hand:
                                    gesture_info = hand['gesture']
                                    st.write(f"**Gesture**: {gesture_info.get('gesture', 'Unknown')}")
                                    st.write(f"**Gesture Confidence**: {gesture_info.get('confidence', 0):.3f}")
                                else:
                                    st.write("**Gesture**: Processing...")
                else:
                    # Compact display for live feed
                    gesture_summary = []
                    for hand in results["hands"]:
                        hand_label = hand['handedness']['label']
                        if 'gesture' in hand:
                            gesture_name = hand['gesture'].get('gesture', 'Unknown')
                            gesture_conf = hand['gesture'].get('confidence', 0)
                            gesture_summary.append(f"{hand_label}: {gesture_name} ({gesture_conf:.2f})")
                        else:
                            gesture_summary.append(f"{hand_label}: Detecting...")
                    
                    if gesture_summary:
                        st.info(" | ".join(gesture_summary))
        else:
            placeholder.warning("👋 No hands detected")
    

    
    def update_stats_display(self, stats_placeholder, history_placeholder, chart_placeholder):
        """Update statistics display with current data."""
        
        # Get current statistics
        frame_count = st.session_state.get('frame_count', 0)
        detection_count = st.session_state.get('detection_count', 0)
        detection_rate = (detection_count / max(frame_count, 1)) * 100
        current_fps = st.session_state.get('current_fps', 0.0)
        
        # Update basic stats
        with stats_placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Total Frames", 
                    frame_count,
                    delta=f"+1" if frame_count > 0 else None
                )
                st.metric(
                    "Detection Rate", 
                    f"{detection_rate:.1f}%",
                    delta=f"{detection_rate-50:.1f}%" if detection_rate != 50 else None,
                    delta_color="normal" if detection_rate > 50 else "inverse"
                )
            
            with col2:
                st.metric(
                    "Detections", 
                    detection_count,
                    delta=f"+1" if detection_count > 0 else None
                )
                
                # FPS with color coding
                fps_color = "🟢" if current_fps > 15 else "🟡" if current_fps > 8 else "🔴"
                st.metric(
                    "FPS", 
                    f"{fps_color} {current_fps:.1f}",
                    delta=f"{current_fps-15:.1f}" if current_fps != 15 else None,
                    delta_color="normal" if current_fps > 15 else "inverse"
                )
        
        # Update gesture history
        gesture_history = st.session_state.get('gesture_history', [])
        
        with history_placeholder.container():
            st.subheader("📝 Recent Gestures")
            
            if gesture_history:
                # Show last 5 gestures for real-time view
                recent_gestures = gesture_history[-5:]
                
                # Create DataFrame for display
                df_data = []
                for entry in recent_gestures:
                    time_ago = int(time.time() - entry['timestamp'])
                    df_data.append({
                        "Time": f"{time_ago}s ago" if time_ago < 60 else time.strftime("%H:%M:%S", time.localtime(entry['timestamp'])),
                        "Gesture": entry['gesture'],
                        "Confidence": f"{entry['confidence']:.2f}",
                        "Hand": entry['handedness']
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_column_width=True, height=200)
                
                # Current gesture status
                if len(recent_gestures) > 0:
                    latest = recent_gestures[-1]
                    seconds_ago = int(time.time() - latest['timestamp'])
                    
                    if seconds_ago < 3:  # Show if within last 3 seconds
                        confidence_emoji = "🟢" if latest['confidence'] > 0.8 else "🟡" if latest['confidence'] > 0.5 else "🔴"
                        st.info(f"**Latest**: {confidence_emoji} {latest['gesture']} ({latest['handedness']}) - {seconds_ago}s ago")
            else:
                st.info("📋 No gestures detected yet")
        
        # Update gesture frequency chart
        with chart_placeholder.container():
            if len(gesture_history) > 3:
                st.subheader("📈 Gesture Frequency (Live)")
                
                # Last 50 gestures for frequency analysis
                recent_for_chart = gesture_history[-50:] if len(gesture_history) > 50 else gesture_history
                
                gesture_counts = {}
                for entry in recent_for_chart:
                    gesture = entry['gesture']
                    gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
                
                # Sort by frequency
                sorted_gestures = sorted(gesture_counts.items(), key=lambda x: x[1], reverse=True)
                
                if sorted_gestures:
                    chart_data = pd.DataFrame([
                        {"Gesture": gesture, "Count": count}
                        for gesture, count in sorted_gestures[:8]  # Top 8 gestures
                    ])
                    
                    st.bar_chart(chart_data.set_index('Gesture'), height=300)
                    
                    # Show percentages
                    total_gestures = sum(gesture_counts.values())
                    top_gesture = sorted_gestures[0]
                    percentage = (top_gesture[1] / total_gestures) * 100
                    st.caption(f"Most frequent: **{top_gesture[0]}** ({percentage:.1f}% of {total_gestures} gestures)")
            else:
                st.info("📊 Frequency chart will appear after detecting more gestures")
    
    def render_status_bar(self):
        """Render status information."""
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            detector_status = "✅ Ready" if st.session_state.get('gesture_detector') and st.session_state.gesture_detector.is_initialized else "❌ Not Ready"
            st.write(f"**Detector**: {detector_status}")
        
        with col2:
            classifier_status = "✅ Ready" if st.session_state.get('gesture_classifier') else "❌ Not Ready"
            st.write(f"**Classifier**: {classifier_status}")
        
        with col3:
            camera_status = "📹 Active" if st.session_state.get('camera_active', False) else "📷 Inactive"
            st.write(f"**Camera**: {camera_status}")
        
        with col4:
            fps = st.session_state.get('current_fps', 0.0)
            fps_status = "🟢 Good" if fps > 15 else "🟡 Low" if fps > 5 else "🔴 Poor"
            st.write(f"**Performance**: {fps_status}")

def main():
    """Main function to run the dashboard."""
    dashboard = HandTrackingTestDashboard()
    dashboard.render()

if __name__ == "__main__":
    main() 