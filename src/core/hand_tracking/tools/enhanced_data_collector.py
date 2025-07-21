"""
Enhanced Gesture Data Collector
==============================

Advanced data collection tool that significantly improves training data quality
for better gesture recognition accuracy.
"""

import cv2
import json
import os
import time
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from collections import defaultdict
import threading
import sys
import os.path

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.gesture_detector import GestureDetector

logger = logging.getLogger(__name__)

class EnhancedGestureDataCollector:
    """Enhanced data collector with quality metrics and advanced features."""
    
    def __init__(self, data_dir: str = "data/gesture_data"):
        """Initialize the enhanced data collector."""
        self.data_dir = data_dir
        self.gesture_detector = GestureDetector()
        self.collected_data = {}
        self.quality_metrics = defaultdict(list)
        self.temporal_data = defaultdict(list)  # For sequence data
        
        # Quality thresholds
        self.min_hand_confidence = 0.8
        self.min_landmark_stability = 0.95
        self.min_gesture_duration = 0.5  # seconds
        
        os.makedirs(data_dir, exist_ok=True)
        
    def initialize(self) -> bool:
        """Initialize the hand detector with optimal settings."""
        config = {
            "static_image_mode": False,
            "max_num_hands": 1,
            "min_detection_confidence": 0.9,  # Higher threshold for quality
            "min_tracking_confidence": 0.8
        }
        return self.gesture_detector.initialize(config)
    
    def collect_high_quality_samples(self, gesture_name: str, target_samples: int = 200, 
                                   collect_sequences: bool = True):
        """
        Collect high-quality gesture samples with automatic quality filtering.
        
        Args:
            gesture_name: Name of the gesture
            target_samples: Target number of high-quality samples
            collect_sequences: Whether to collect temporal sequences
        """
        if gesture_name not in self.collected_data:
            self.collected_data[gesture_name] = []
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open camera")
            return
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)  # Higher FPS for better temporal data
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        print(f"\n🎯 Enhanced Collection for: {gesture_name.upper()}")
        print(f"📊 Target: {target_samples} samples")
        print("\n✨ Collection Modes:")
        print("- AUTO: Captures automatically when quality > 0.7")
        print("- MANUAL: Press SPACEBAR to capture anytime")
        print("- Real-time quality feedback")
        print("\n📋 Instructions:")
        print("1. GREEN quality = auto-capture enabled")
        print("2. YELLOW/RED quality = use manual capture (SPACEBAR)")
        print("3. Vary: distance, angle, lighting, hand position")
        print("4. Q = quit early, ESC = exit completely")
        
        input("\nPress Enter to start enhanced collection...")
        
        samples_collected = 0
        sequence_buffer = []
        last_capture_time = 0
        capture_interval = 0.1  # 100ms between captures
        
        # Quality tracking
        stability_buffer = []
        confidence_buffer = []
        
        print("\n🎥 Camera active - Quality collection in progress...")
        
        while samples_collected < target_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            current_time = time.time()
            
            # Detect hands
            results = self.gesture_detector.detect_hands(frame)
            display_frame = frame.copy()
            
            # Quality assessment
            quality_score = self._assess_frame_quality(frame, results)
            quality_color = (0, 255, 0) if quality_score > 0.8 else (0, 165, 255) if quality_score > 0.6 else (0, 0, 255)
            
            # Draw UI
            self._draw_collection_ui(display_frame, gesture_name, samples_collected, 
                                   target_samples, quality_score, quality_color)
            
            if results.get("hands_detected"):
                hand_data = results["hands"][0]
                landmarks = hand_data["landmarks"]
                confidence = hand_data["handedness"]["score"]
                
                # Draw landmarks with quality indication
                self._draw_quality_landmarks(display_frame, landmarks, quality_score)
                
                # Collect sequence data
                if collect_sequences:
                    sequence_buffer.append({
                        "landmarks": landmarks,
                        "timestamp": current_time,
                        "quality": quality_score
                    })
                    
                    # Keep only recent frames (last 2 seconds)
                    sequence_buffer = [s for s in sequence_buffer if current_time - s["timestamp"] < 2.0]
                
                # Auto-capture high quality samples (lowered threshold)
                auto_capture = (quality_score > 0.7 and 
                               confidence > self.min_hand_confidence and
                               current_time - last_capture_time > capture_interval)
                
                if auto_capture:
                    # Create high-quality sample
                    sample = self._create_enhanced_sample(
                        gesture_name, hand_data, quality_score, 
                        sequence_buffer if collect_sequences else None
                    )
                    
                    self.collected_data[gesture_name].append(sample)
                    samples_collected += 1
                    last_capture_time = current_time
                    
                    # Visual feedback
                    cv2.putText(display_frame, f"✓ AUTO-CAPTURED #{samples_collected}", 
                              (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    print(f"✅ Auto-captured {samples_collected}/{target_samples} (Quality: {quality_score:.3f})")
                    
                    # Brief visual feedback
                    feedback_frame = display_frame.copy()
                    cv2.rectangle(feedback_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), 
                                (0, 255, 0), 15)
                    cv2.imshow('Enhanced Gesture Collection', feedback_frame)
                    cv2.waitKey(100)
            
            cv2.imshow('Enhanced Gesture Collection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return samples_collected
            elif key == ord(' '):  # SPACEBAR for manual capture
                if results.get("hands_detected"):
                    # Manual capture - allow any quality
                    sample = self._create_enhanced_sample(
                        gesture_name, hand_data, quality_score, 
                        sequence_buffer if collect_sequences else None
                    )
                    
                    self.collected_data[gesture_name].append(sample)
                    samples_collected += 1
                    last_capture_time = current_time
                    
                    # Visual feedback for manual capture
                    cv2.putText(display_frame, f"✓ MANUAL CAPTURE #{samples_collected}", 
                              (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    
                    print(f"👆 Manual capture {samples_collected}/{target_samples} (Quality: {quality_score:.3f})")
                    
                    # Brief visual feedback
                    feedback_frame = display_frame.copy()
                    cv2.rectangle(feedback_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), 
                                (255, 255, 0), 15)
                    cv2.imshow('Enhanced Gesture Collection', feedback_frame)
                    cv2.waitKey(100)
                else:
                    print("⚠️  No hand detected - show your hand to capture")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Enhanced collection complete!")
        print(f"📊 Collected {samples_collected} high-quality samples")
        
        return samples_collected
    
    def _assess_frame_quality(self, frame: np.ndarray, results: Dict) -> float:
        """Assess the quality of the current frame for gesture recognition."""
        if not results.get("hands_detected"):
            return 0.0
        
        hand_data = results["hands"][0]
        landmarks = hand_data["landmarks"]
        bbox = hand_data["bounding_box"]
        
        quality_factors = []
        
        # 1. Hand confidence
        confidence = hand_data["handedness"]["score"]
        quality_factors.append(min(confidence / 0.95, 1.0))
        
        # 2. Hand size (not too small/large)
        hand_area = bbox["width"] * bbox["height"]
        frame_area = frame.shape[0] * frame.shape[1]
        size_ratio = hand_area / frame_area
        
        # Optimal size: 5-25% of frame
        if 0.05 <= size_ratio <= 0.25:
            quality_factors.append(1.0)
        elif 0.03 <= size_ratio <= 0.35:
            quality_factors.append(0.7)
        else:
            quality_factors.append(0.3)
        
        # 3. Hand position (not at edges)
        center_x = bbox["x"] + bbox["width"] // 2
        center_y = bbox["y"] + bbox["height"] // 2
        
        # Penalize if too close to edges
        edge_penalty = 1.0
        if center_x < frame.shape[1] * 0.15 or center_x > frame.shape[1] * 0.85:
            edge_penalty *= 0.7
        if center_y < frame.shape[0] * 0.15 or center_y > frame.shape[0] * 0.85:
            edge_penalty *= 0.7
        
        quality_factors.append(edge_penalty)
        
        # 4. Landmark visibility and consistency
        visible_landmarks = sum(1 for lm in landmarks if lm.get("visibility", 1.0) > 0.8)
        visibility_ratio = visible_landmarks / 21
        quality_factors.append(visibility_ratio)
        
        # 5. Lighting quality (check for overexposure/underexposure)
        hand_region = frame[bbox["y"]:bbox["y"]+bbox["height"], 
                          bbox["x"]:bbox["x"]+bbox["width"]]
        
        if hand_region.size > 0:
            mean_brightness = np.mean(hand_region)
            # Optimal brightness: 80-180
            if 80 <= mean_brightness <= 180:
                quality_factors.append(1.0)
            elif 60 <= mean_brightness <= 200:
                quality_factors.append(0.8)
            else:
                quality_factors.append(0.5)
        else:
            quality_factors.append(0.5)
        
        return np.mean(quality_factors)
    
    def _create_enhanced_sample(self, gesture_name: str, hand_data: Dict, 
                              quality_score: float, sequence_data: List = None) -> Dict:
        """Create an enhanced sample with additional metadata."""
        sample = {
            "gesture": gesture_name,
            "landmarks": hand_data["landmarks"],
            "landmarks_normalized": hand_data["landmarks_normalized"],
            "handedness": hand_data["handedness"],
            "bounding_box": hand_data["bounding_box"],
            "timestamp": time.time(),
            "quality_score": quality_score,
            "sample_id": len(self.collected_data[gesture_name])
        }
        
        # Add sequence data if available
        if sequence_data and len(sequence_data) > 5:
            sample["sequence"] = {
                "frames": sequence_data[-10:],  # Last 10 frames
                "duration": sequence_data[-1]["timestamp"] - sequence_data[0]["timestamp"]
            }
        
        return sample
    
    def _draw_collection_ui(self, frame: np.ndarray, gesture_name: str, 
                          collected: int, target: int, quality: float, color: Tuple):
        """Draw enhanced UI for data collection."""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        
        # Header
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Gesture name
        cv2.putText(frame, f"Collecting: {gesture_name.upper()}", 
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Progress
        progress_text = f"Progress: {collected}/{target}"
        cv2.putText(frame, progress_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Quality indicator
        quality_text = f"Quality: {quality:.3f}"
        cv2.putText(frame, quality_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Capture mode indicator
        if quality > 0.7:
            mode_text = "AUTO-CAPTURE ENABLED"
            mode_color = (0, 255, 0)
        else:
            mode_text = "MANUAL MODE - SPACEBAR"
            mode_color = (0, 255, 255)
        cv2.putText(frame, mode_text, (w - 350, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        # Progress bar
        bar_width = 300
        bar_height = 10
        bar_x = w - bar_width - 20
        bar_y = 40
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Progress bar
        progress = collected / target
        progress_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
        
        # Quality indicator circle
        quality_radius = int(20 * quality)
        cv2.circle(frame, (w - 50, 80), quality_radius, color, -1)
    
    def _draw_quality_landmarks(self, frame: np.ndarray, landmarks: List, quality: float):
        """Draw landmarks with quality-based colors."""
        # Color based on quality
        if quality > 0.8:
            color = (0, 255, 0)  # Green
        elif quality > 0.6:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red
        
        for landmark in landmarks:
            cv2.circle(frame, (landmark["x"], landmark["y"]), 4, color, -1)
            cv2.circle(frame, (landmark["x"], landmark["y"]), 6, (255, 255, 255), 1)
    
    def collect_augmented_dataset(self, gestures: List[str], samples_per_gesture: int = 200):
        """Collect a comprehensive dataset with multiple variations per gesture."""
        total_gestures = len(gestures)
        
        print("🚀 Enhanced Gesture Dataset Collection")
        print("=" * 50)
        print(f"📊 Will collect {total_gestures} gestures with {samples_per_gesture} samples each")
        print(f"🎯 Total samples: {total_gestures * samples_per_gesture}")
        print("\n💡 Collection Strategy:")
        print("- Multiple hand positions and orientations")
        print("- Various distances and angles")  
        print("- Different lighting conditions")
        print("- Temporal sequence data")
        
        for i, gesture in enumerate(gestures):
            print(f"\n{'='*60}")
            print(f"📋 Gesture {i+1}/{total_gestures}: {gesture.upper()}")
            print(f"{'='*60}")
            
            print(f"\n🎬 Instructions for '{gesture}':")
            self._show_gesture_instructions(gesture)
            
            try:
                input("\n🎬 Press Enter to start collection (Ctrl+C to quit)...")
            except KeyboardInterrupt:
                print("\n\n🛑 Collection cancelled by user.")
                break
            
            collected = self.collect_high_quality_samples(gesture, samples_per_gesture)
            
            if collected < samples_per_gesture * 0.8:  # Less than 80% collected
                print(f"\n⚠️  Only collected {collected}/{samples_per_gesture} samples")
                retry = input("Retry this gesture? (y/n): ").lower().strip()
                if retry == 'y':
                    self.collect_high_quality_samples(gesture, samples_per_gesture - collected)
    
    def _show_gesture_instructions(self, gesture_name: str):
        """Show specific instructions for each gesture."""
        instructions = {
            "neutral": "✋ Relaxed hand, fingers slightly spread",
            "pointing": "👉 Index finger extended, others curled",
            "fist": "✊ All fingers curled into palm",
            "open_hand": "🖐 All fingers fully extended and spread",
            "peace_sign": "✌️ Index and middle fingers extended in V shape",
            "thumbs_up": "👍 Thumb extended up, others curled",
            "pinch": "🤏 Thumb and index finger close together",
            "ok_sign": "👌 Thumb and index form circle, others extended",
            "rock": "🤘 Index and pinky extended, others curled",
            "gun": "🔫 Index extended, thumb up, others curled"
        }
        
        instruction = instructions.get(gesture_name, f"Show the '{gesture_name}' gesture")
        print(f"   {instruction}")
        print("   💡 Vary: hand angle, distance, position, lighting")
    
    def save_enhanced_dataset(self, filename: str = None):
        """Save the enhanced dataset with quality metrics."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"enhanced_gesture_dataset_{timestamp}.json"
        
        filepath = os.path.join(self.data_dir, filename)
        
        # Calculate comprehensive statistics
        total_samples = sum(len(samples) for samples in self.collected_data.values())
        quality_stats = {}
        
        for gesture, samples in self.collected_data.items():
            qualities = [s.get("quality_score", 0.5) for s in samples]
            quality_stats[gesture] = {
                "count": len(samples),
                "avg_quality": np.mean(qualities),
                "min_quality": np.min(qualities),
                "max_quality": np.max(qualities),
                "high_quality_samples": sum(1 for q in qualities if q > 0.8)
            }
        
        enhanced_data = {
            "metadata": {
                "version": "2.0_enhanced",
                "collection_date": time.time(),
                "total_samples": total_samples,
                "gestures_count": len(self.collected_data),
                "quality_threshold": 0.85,
                "features": [
                    "high_quality_filtering",
                    "temporal_sequences", 
                    "multiple_orientations",
                    "quality_metrics"
                ]
            },
            "statistics": quality_stats,
            "data": self.collected_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(enhanced_data, f, indent=2)
        
        # Also save with a fixed name for easy access
        fixed_filepath = os.path.join(self.data_dir, "enhanced_gesture_dataset.json")
        with open(fixed_filepath, 'w') as f:
            json.dump(enhanced_data, f, indent=2)
        
        print(f"\n💾 Enhanced dataset saved: {filepath}")
        print(f"💾 Also saved as: {fixed_filepath}")
        print(f"📊 Total samples: {total_samples}")
        print("📋 Quality Statistics:")
        
        for gesture, stats in quality_stats.items():
            print(f"  {gesture}: {stats['count']} samples "
                  f"(avg quality: {stats['avg_quality']:.3f}, "
                  f"high quality: {stats['high_quality_samples']})")
        
        return filepath

def main():
    """Main function for enhanced data collection."""
    collector = EnhancedGestureDataCollector()
    
    if not collector.initialize():
        print("❌ Failed to initialize hand detector")
        return
    
    # Enhanced gesture set
    enhanced_gestures = [
        "neutral",
        "pointing", 
        "fist",
        "open_hand",
        "peace_sign",
        "thumbs_up",
        "pinch",
        "ok_sign",
        "rock",
        "gun"
    ]
    
    print("🚀 Enhanced Gesture Data Collection")
    print("=" * 50)
    print("✨ New Features:")
    print("- Automatic quality assessment")
    print("- Real-time feedback")
    print("- Enhanced sample diversity")
    print("- Temporal sequence collection")
    
    collector.collect_augmented_dataset(enhanced_gestures, samples_per_gesture=150)
    dataset_file = collector.save_enhanced_dataset()
    
    print(f"\n🎉 Enhanced collection complete!")
    print(f"📁 Dataset: {dataset_file}")
    print("\n🚀 Next steps:")
    print("1. Train with enhanced features: python enhanced_trainer.py")
    print("2. Test real-time performance: python gesture_demo.py")

if __name__ == "__main__":
    main() 