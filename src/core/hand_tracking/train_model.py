"""
Model Training Script
=====================

Train gesture recognition model with your collected data.
Run this from the hand_tracking directory: python train_model.py
"""

import os
import json
import numpy as np
from gesture_classifier import GestureClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train and evaluate gesture recognition models."""
    
    def __init__(self, data_file: str = "./gesture_data/training_data.json"):
        """Initialize the trainer with data file."""
        self.data_file = data_file
        self.classifiers = {}
        
    def check_data_file(self):
        """Check if training data exists."""
        if not os.path.exists(self.data_file):
            print(f"❌ Training data not found: {self.data_file}")
            print("\nPlease run data collection first:")
            print("  python data_collector.py")
            return False
        
        # Load and check data
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            
            stats = data.get('statistics', {})
            total_samples = stats.get('total_samples', 0)
            gestures = stats.get('gestures', {})
            
            print(f"✅ Training data found: {self.data_file}")
            print(f"📊 Total samples: {total_samples}")
            print("📋 Gestures and sample counts:")
            for gesture, count in gestures.items():
                print(f"  {gesture}: {count} samples")
            
            # Check minimum samples
            min_samples = min(gestures.values()) if gestures else 0
            if min_samples < 10:
                print(f"⚠️  Warning: Some gestures have few samples (min: {min_samples})")
                print("   Consider collecting more data for better performance")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            return False
    
    def train_model(self, model_type: str = 'knn') -> GestureClassifier:
        """Train a gesture classifier with the collected data."""
        print(f"\n🤖 Training {model_type.upper()} model...")
        
        # Create classifier
        classifier = GestureClassifier(model_type=model_type)
        
        # Train with collected data
        start_time = time.time()
        classifier.train(data_file=self.data_file)
        training_time = time.time() - start_time
        
        print(f"✅ Model trained in {training_time:.2f} seconds")
        
        # Store for comparison
        self.classifiers[model_type] = classifier
        
        return classifier
    
    def evaluate_model(self, classifier: GestureClassifier, model_name: str):
        """Evaluate model performance with cross-validation."""
        print(f"\n📊 Evaluating {model_name} model...")
        
        # Load data for evaluation
        features, labels = classifier.load_real_training_data(self.data_file)
        
        # Cross-validation
        cv_scores = cross_val_score(classifier.model, 
                                   classifier.scaler.transform(features), 
                                   labels, cv=5)
        
        print(f"📈 Cross-validation scores: {cv_scores}")
        print(f"🎯 Mean accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Test with some samples
        print(f"\n🧪 Testing {model_name} predictions...")
        gesture_names = list(classifier.gesture_labels.values())
        
        # Test each gesture type
        for gesture_name in gesture_names:
            if hasattr(classifier, '_generate_synthetic_landmarks'):
                test_landmarks = classifier._generate_synthetic_landmarks(gesture_name)
                result = classifier.predict_gesture(test_landmarks)
                confidence = result['confidence']
                predicted = result['gesture']
                
                status = "✅" if predicted == gesture_name else "❌"
                print(f"  {status} {gesture_name}: predicted '{predicted}' (confidence: {confidence:.3f})")
    
    def compare_models(self):
        """Compare different model types."""
        print("\n🔄 Comparing different models...")
        
        model_types = ['knn', 'rf']  # KNN and Random Forest
        results = {}
        
        for model_type in model_types:
            print(f"\n--- Training {model_type.upper()} ---")
            try:
                classifier = self.train_model(model_type)
                
                # Quick evaluation
                features, labels = classifier.load_real_training_data(self.data_file)
                cv_scores = cross_val_score(classifier.model, 
                                           classifier.scaler.transform(features), 
                                           labels, cv=3)
                
                results[model_type] = {
                    'mean_accuracy': cv_scores.mean(),
                    'std_accuracy': cv_scores.std(),
                    'classifier': classifier
                }
                
                print(f"✅ {model_type.upper()} accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                
            except Exception as e:
                print(f"❌ Error training {model_type}: {e}")
                results[model_type] = None
        
        # Find best model
        best_model = None
        best_accuracy = 0
        
        print(f"\n🏆 Model Comparison Results:")
        for model_type, result in results.items():
            if result:
                accuracy = result['mean_accuracy']
                std = result['std_accuracy']
                print(f"  {model_type.upper()}: {accuracy:.3f} (+/- {std * 2:.3f})")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = (model_type, result['classifier'])
        
        if best_model:
            model_name, classifier = best_model
            print(f"\n🥇 Best model: {model_name.upper()} with {best_accuracy:.3f} accuracy")
            return classifier, model_name
        else:
            print("\n❌ No models were successfully trained")
            return None, None
    
    def save_model(self, classifier: GestureClassifier, model_name: str):
        """Save the trained model."""
        model_path = f"./gesture_data/trained_model_{model_name}.pkl"
        
        try:
            classifier.save_model(model_path)
            print(f"✅ Model saved to: {model_path}")
            return model_path
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return None
    
    def test_real_time_performance(self, classifier: GestureClassifier):
        """Test real-time prediction performance."""
        print(f"\n⚡ Testing real-time performance...")
        
        # Generate test data
        test_count = 100
        start_time = time.time()
        
        for _ in range(test_count):
            # Use random gesture for testing
            gesture_name = np.random.choice(list(classifier.gesture_labels.values()))
            test_landmarks = classifier._generate_synthetic_landmarks(gesture_name)
            result = classifier.predict_gesture(test_landmarks)
        
        end_time = time.time()
        total_time = end_time - start_time
        predictions_per_second = test_count / total_time
        
        print(f"📊 Performance: {predictions_per_second:.1f} predictions/second")
        print(f"🎯 Average prediction time: {(total_time/test_count)*1000:.2f}ms")
        
        if predictions_per_second > 30:
            print("✅ Real-time capable (>30 FPS)")
        else:
            print("⚠️  May struggle with real-time performance")

def main():
    """Main training function."""
    print("🤖 Gesture Recognition Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Check if data exists
    if not trainer.check_data_file():
        return
    
    print(f"\n🚀 Starting training process...")
    
    # Option 1: Train single model
    print(f"\n📋 Choose training option:")
    print("1. Train single model (fast)")
    print("2. Compare multiple models (slower but finds best)")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
    except KeyboardInterrupt:
        print("\n\nTraining cancelled.")
        return
    
    if choice == "1":
        # Train single KNN model
        classifier = trainer.train_model('knn')
        trainer.evaluate_model(classifier, 'KNN')
        trainer.test_real_time_performance(classifier)
        model_path = trainer.save_model(classifier, 'knn')
        
    elif choice == "2":
        # Compare models and use best
        best_classifier, best_model_name = trainer.compare_models()
        
        if best_classifier:
            trainer.evaluate_model(best_classifier, best_model_name)
            trainer.test_real_time_performance(best_classifier)
            model_path = trainer.save_model(best_classifier, best_model_name)
        else:
            print("❌ No models were successfully trained")
            return
    
    else:
        print("❌ Invalid choice. Using default KNN model.")
        classifier = trainer.train_model('knn')
        trainer.evaluate_model(classifier, 'KNN')
        model_path = trainer.save_model(classifier, 'knn')
    
    # Final instructions
    print(f"\n🎉 Training complete!")
    print(f"\nNext steps:")
    print("1. Test your model with real-time recognition:")
    print("   python gesture_classification_demo.py")
    print("2. Test individual gestures:")
    print("   python test_gesture_classifier.py")
    print("3. Collect more data to improve performance:")
    print("   python data_collector.py")

if __name__ == "__main__":
    main() 