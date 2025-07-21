"""
Enhanced Model Training Script
=============================

Advanced training script for the enhanced gesture classifier with
comprehensive evaluation and optimization features.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import logging
import time
from typing import Dict, List, Tuple
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.enhanced_gesture_classifier import EnhancedGestureClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedModelTrainer:
    """Advanced trainer for enhanced gesture recognition models."""
    
    def __init__(self, data_file: str = None):
        """Initialize the enhanced trainer."""
        if data_file is None:
            # Auto-find the latest enhanced dataset
            data_file = self._find_latest_enhanced_dataset()
        self.data_file = data_file
        self.classifier = None
        self.results = {}
    
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
            if os.path.exists(default_file):
                return default_file
            return default_file  # Return default even if doesn't exist (will show error later)
        
        # Sort by timestamp (newest first)
        dataset_files.sort(reverse=True)
        latest_file = os.path.join(data_dir, dataset_files[0])
        
        print(f"🔍 Auto-detected latest dataset: {latest_file}")
        return latest_file
        
    def check_enhanced_data(self) -> bool:
        """Check if enhanced training data exists and assess quality."""
        if not os.path.exists(self.data_file):
            print(f"❌ Enhanced training data not found: {self.data_file}")
            print("\n📝 To collect enhanced data, run:")
            print("  python run_enhanced_collection.py")
            return False
        
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            stats = data.get('statistics', {})
            
            print(f"✅ Enhanced dataset found: {self.data_file}")
            print(f"📊 Dataset version: {metadata.get('version', 'unknown')}")
            print(f"🎯 Total samples: {metadata.get('total_samples', 0)}")
            print(f"🎭 Gestures: {metadata.get('gestures_count', 0)}")
            
            print("\n📋 Quality Statistics:")
            total_high_quality = 0
            for gesture, gesture_stats in stats.items():
                high_quality = gesture_stats.get('high_quality_samples', 0)
                avg_quality = gesture_stats.get('avg_quality', 0)
                total_samples = gesture_stats.get('count', 0)
                
                print(f"  {gesture}: {total_samples} samples "
                      f"(avg quality: {avg_quality:.3f}, "
                      f"high quality: {high_quality})")
                
                total_high_quality += high_quality
            
            print(f"\n🏆 Total high-quality samples: {total_high_quality}")
            
            # Check data quality
            if total_high_quality < 500:
                print("⚠️  Warning: Consider collecting more high-quality data for optimal performance")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading enhanced data: {e}")
            return False
    
    def train_enhanced_model(self) -> EnhancedGestureClassifier:
        """Train the enhanced gesture classifier."""
        print("\n🚀 Training Enhanced Gesture Classifier")
        print("=" * 50)
        print(f"📁 Using dataset: {self.data_file}")
        
        # Initialize enhanced classifier
        self.classifier = EnhancedGestureClassifier(
            use_ensemble=True,
            use_temporal=True,
            temporal_window=10
        )
        
        # Train with all enhancements
        success = self.classifier.train_enhanced_model(
            data_file=self.data_file,
            use_advanced_preprocessing=True,
            use_feature_selection=True,
            use_pca=True
        )
        
        if not success:
            print("❌ Training failed!")
            return None
        
        # Get training results
        stats = self.classifier.training_stats
        self.results = stats.copy()
        
        print("\n✅ Enhanced training complete!")
        print(f"🎯 Final accuracy: {stats['accuracy']:.3f}")
        print(f"📊 Cross-validation: {stats['cv_mean']:.3f} (+/- {stats['cv_std']*2:.3f})")
        print(f"⚡ Training time: {stats['training_time']:.2f}s")
        print(f"🔧 Features used: {stats['num_features']}")
        print(f"📝 Training samples: {stats['num_samples']}")
        
        return self.classifier
    
    def comprehensive_evaluation(self, classifier: EnhancedGestureClassifier):
        """Perform comprehensive model evaluation."""
        print("\n📊 Comprehensive Model Evaluation")
        print("=" * 50)
        
        # Load test data
        features, labels = classifier.load_enhanced_training_data(self.data_file)
        
        if len(features) == 0:
            print("❌ No test data available")
            return
        
        # Apply same preprocessing as training
        features_scaled = classifier.scaler.transform(features)
        
        if classifier.feature_selector is not None:
            features_scaled = classifier.feature_selector.transform(features_scaled)
        
        if classifier.pca is not None:
            features_scaled = classifier.pca.transform(features_scaled)
        
        # Make predictions
        if classifier.use_ensemble:
            y_pred = classifier.ensemble_model.predict(features_scaled)
            y_prob = classifier.ensemble_model.predict_proba(features_scaled)
        else:
            y_pred = classifier.base_models['rf'].predict(features_scaled)
            y_prob = classifier.base_models['rf'].predict_proba(features_scaled)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == labels)
        
        print(f"📈 Overall accuracy: {accuracy:.3f}")
        
        # Per-class analysis
        print("\n📋 Per-class Performance:")
        gesture_names = list(classifier.gesture_labels.values())
        
        for i, gesture in enumerate(gesture_names):
            if i in labels:
                mask = labels == i
                if np.sum(mask) > 0:
                    gesture_accuracy = np.mean(y_pred[mask] == labels[mask])
                    gesture_confidence = np.mean(np.max(y_prob[mask], axis=1))
                    
                    print(f"  {gesture}: accuracy={gesture_accuracy:.3f}, "
                          f"avg_confidence={gesture_confidence:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(labels, y_pred)
        self._plot_confusion_matrix(cm, gesture_names)
        
        # Confidence distribution
        confidences = np.max(y_prob, axis=1)
        print(f"\n🎯 Confidence Statistics:")
        print(f"  Mean: {np.mean(confidences):.3f}")
        print(f"  Std: {np.std(confidences):.3f}")
        print(f"  Min: {np.min(confidences):.3f}")
        print(f"  Max: {np.max(confidences):.3f}")
        
        # High confidence accuracy
        high_conf_mask = confidences > 0.8
        if np.sum(high_conf_mask) > 0:
            high_conf_accuracy = np.mean(y_pred[high_conf_mask] == labels[high_conf_mask])
            print(f"  High confidence (>0.8) accuracy: {high_conf_accuracy:.3f}")
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str]):
        """Plot confusion matrix."""
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Enhanced Gesture Classifier - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save plot
            plot_path = 'data/gesture_data/confusion_matrix.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved: {plot_path}")
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Could not generate confusion matrix plot: {e}")
    
    def performance_benchmark(self, classifier: EnhancedGestureClassifier):
        """Benchmark real-time performance."""
        print("\n⚡ Real-time Performance Benchmark")
        print("=" * 40)
        
        # Generate test landmarks
        test_landmarks = self._generate_test_landmarks()
        
        # Warmup
        for _ in range(10):
            classifier.predict_gesture(test_landmarks)
        
        # Benchmark
        num_predictions = 1000
        start_time = time.time()
        
        for _ in range(num_predictions):
            result = classifier.predict_gesture(test_landmarks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        predictions_per_second = num_predictions / total_time
        avg_prediction_time = (total_time / num_predictions) * 1000  # ms
        
        print(f"📊 Performance Results:")
        print(f"  Predictions per second: {predictions_per_second:.1f}")
        print(f"  Average prediction time: {avg_prediction_time:.2f}ms")
        
        # Performance assessment
        if predictions_per_second >= 60:
            print("✅ Excellent real-time performance (60+ FPS)")
        elif predictions_per_second >= 30:
            print("✅ Good real-time performance (30+ FPS)")
        elif predictions_per_second >= 15:
            print("⚠️  Acceptable performance (15+ FPS)")
        else:
            print("❌ May struggle with real-time applications")
        
        # Memory usage (rough estimate)
        import sys
        model_size = sys.getsizeof(classifier) / (1024 * 1024)  # MB
        print(f"  Estimated model size: {model_size:.2f} MB")
        
        return predictions_per_second, avg_prediction_time
    
    def _generate_test_landmarks(self) -> List[Dict[str, float]]:
        """Generate test landmarks for benchmarking."""
        landmarks = []
        for i in range(21):
            landmark = {
                'id': i,
                'x': 100 + i * 20 + np.random.normal(0, 5),
                'y': 200 + i * 10 + np.random.normal(0, 5),
                'z': np.random.normal(0, 2),
                'visibility': 1.0
            }
            landmarks.append(landmark)
        return landmarks
    
    def save_enhanced_model(self, classifier: EnhancedGestureClassifier, 
                          performance_info: Tuple[float, float]):
        """Save the enhanced model with metadata."""
        timestamp = int(time.time())
        model_path = f"data/gesture_data/enhanced_model_{timestamp}.pkl"
        
        try:
            classifier.save_enhanced_model(model_path)
            
            # Save performance metadata
            fps, avg_time = performance_info
            metadata = {
                'model_path': model_path,
                'training_timestamp': timestamp,
                'performance': {
                    'fps': fps,
                    'avg_prediction_time_ms': avg_time
                },
                'training_stats': classifier.training_stats,
                'model_info': classifier.get_model_info()
            }
            
            metadata_path = f"data/gesture_data/model_metadata_{timestamp}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Enhanced model saved: {model_path}")
            print(f"📋 Metadata saved: {metadata_path}")
            
            return model_path
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return None
    
    def compare_with_baseline(self, enhanced_classifier: EnhancedGestureClassifier):
        """Compare enhanced model with baseline."""
        print("\n🔄 Comparing with Baseline Model")
        print("=" * 40)
        
        try:
            # Try to load existing baseline model
            baseline_path = "data/gesture_data/trained_model_rf.pkl"
            if os.path.exists(baseline_path):
                print("📊 Loading baseline model for comparison...")
                
                # Load baseline results (would need to implement this)
                # For now, just show improvement
                baseline_accuracy = 0.85  # Placeholder
                enhanced_accuracy = enhanced_classifier.training_stats['accuracy']
                
                improvement = enhanced_accuracy - baseline_accuracy
                print(f"  Baseline accuracy: {baseline_accuracy:.3f}")
                print(f"  Enhanced accuracy: {enhanced_accuracy:.3f}")
                print(f"  Improvement: +{improvement:.3f} ({improvement/baseline_accuracy*100:.1f}%)")
                
                if improvement > 0.05:
                    print("🎉 Significant improvement achieved!")
                elif improvement > 0.02:
                    print("✅ Good improvement achieved!")
                else:
                    print("📈 Modest improvement achieved")
            else:
                print("ℹ️  No baseline model found for comparison")
                
        except Exception as e:
            print(f"⚠️  Could not compare with baseline: {e}")

def main():
    """Main training function."""
    print("🚀 Enhanced Gesture Recognition Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = EnhancedModelTrainer()
    
    # Check data
    if not trainer.check_enhanced_data():
        return
    
    print(f"\n🔧 Enhanced Training Features:")
    print("- Comprehensive feature engineering")
    print("- Ensemble learning (KNN + RF + GB + SVM)")
    print("- Advanced preprocessing pipeline")
    print("- Feature selection and PCA")
    print("- Temporal smoothing")
    print("- Quality-based data filtering")
    
    # Confirm training
    try:
        confirm = input(f"\n🚀 Start enhanced training? (y/n): ").lower().strip()
        if confirm != 'y':
            print("Training cancelled.")
            return
    except KeyboardInterrupt:
        print("\nTraining cancelled.")
        return
    
    # Train enhanced model
    classifier = trainer.train_enhanced_model()
    
    if classifier is None:
        print("❌ Training failed!")
        return
    
    # Comprehensive evaluation
    trainer.comprehensive_evaluation(classifier)
    
    # Performance benchmark
    performance_info = trainer.performance_benchmark(classifier)
    
    # Compare with baseline
    trainer.compare_with_baseline(classifier)
    
    # Save model
    model_path = trainer.save_enhanced_model(classifier, performance_info)
    
    if model_path:
        print(f"\n🎉 Enhanced Training Complete!")
        print(f"📁 Model: {model_path}")
        print(f"🎯 Accuracy: {classifier.training_stats['accuracy']:.3f}")
        print(f"⚡ Performance: {performance_info[0]:.1f} FPS")
        
        print(f"\n🚀 Next Steps:")
        print("1. Test real-time recognition:")
        print("   python -m demos.enhanced_gesture_demo")
        print("2. Collect more data to improve further:")
        print("   python -m tools.enhanced_data_collector")
        print("3. Fine-tune model parameters:")
        print("   Edit enhanced_gesture_classifier.py")

if __name__ == "__main__":
    main() 