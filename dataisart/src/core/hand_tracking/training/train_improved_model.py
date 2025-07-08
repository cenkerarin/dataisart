"""
Improved Gesture Model Training
===============================

Train gesture classifier with real data and advanced techniques.
"""

import os
import json
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from ..core.gesture_classifier import GestureClassifier
from ..data.data_collector import GestureDataCollector
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedGestureTrainer:
    """Advanced trainer for gesture classification models."""
    
    def __init__(self, data_dir: str = "../data/gestures"):
        """Initialize the trainer."""
        self.data_dir = data_dir
        self.results = {}
        
    def collect_training_data(self):
        """Collect real training data using the data collector."""
        print("Starting training data collection...")
        collector = GestureDataCollector(self.data_dir)
        
        if not collector.initialize():
            raise RuntimeError("Failed to initialize data collector")
        
        # Define comprehensive gesture set
        gestures = [
            "neutral",      # Relaxed hand
            "pointing",     # Index finger extended
            "fist",         # Closed fist
            "open_hand",    # All fingers extended
            "peace_sign",   # Index and middle extended
            "thumbs_up",    # Thumb extended, others closed
            "pinch",        # Thumb and index close together
            "ok_sign",      # Thumb and index forming circle
            "rock_sign",    # Index and pinky extended
            "three_fingers" # Thumb, index, middle extended
        ]
        
        print(f"Will collect data for {len(gestures)} gestures")
        print("Each gesture will need 40-50 samples for good training")
        
        collector.collect_all_gestures(gestures, samples_per_gesture=40)
        collector.save_data("training_data.json")
        
        return os.path.join(self.data_dir, "training_data.json")
    
    def train_multiple_models(self, data_file: str):
        """Train and compare multiple ML models."""
        print("Training multiple models...")
        
        # Load data
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Prepare data for training
        classifier = GestureClassifier()
        features, labels = classifier.load_real_training_data(data_file)
        
        print(f"Loaded {len(features)} samples with {features.shape[1]} features")
        print(f"Classes: {set(labels)}")
        
        # Define models to test
        models = {
            'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        # Train and evaluate each model
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, features, labels, cv=5, scoring='accuracy')
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            print(f"{name} CV Accuracy: {mean_score:.3f} (+/- {std_score * 2:.3f})")
            
            self.results[name] = {
                'cv_scores': cv_scores,
                'mean_score': mean_score,
                'std_score': std_score,
                'model': model
            }
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = (name, model)
        
        print(f"\nBest model: {best_model[0]} with accuracy: {best_score:.3f}")
        return best_model
    
    def hyperparameter_tuning(self, data_file: str, model_type: str = 'Random Forest'):
        """Perform hyperparameter tuning for the best model."""
        print(f"Performing hyperparameter tuning for {model_type}...")
        
        classifier = GestureClassifier()
        features, labels = classifier.load_real_training_data(data_file)
        
        if model_type == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
            
        elif model_type == 'SVM':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'poly']
            }
            model = SVC(probability=True, random_state=42)
            
        elif model_type == 'KNN':
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
            model = KNeighborsClassifier()
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(features, labels)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def create_optimized_classifier(self, data_file: str, best_params: dict, model_type: str):
        """Create an optimized gesture classifier."""
        print("Creating optimized classifier...")
        
        # Create classifier with optimized parameters
        if model_type == 'Random Forest':
            optimized_model = RandomForestClassifier(**best_params, random_state=42)
        elif model_type == 'SVM':
            optimized_model = SVC(**best_params, probability=True, random_state=42)
        elif model_type == 'KNN':
            optimized_model = KNeighborsClassifier(**best_params)
        
        # Create custom classifier
        classifier = GestureClassifier(model_type='custom')
        classifier.model = optimized_model
        
        # Train with real data
        classifier.train(data_file=data_file)
        
        return classifier
    
    def evaluate_model(self, classifier: GestureClassifier, data_file: str):
        """Detailed evaluation of the trained model."""
        print("Evaluating model performance...")
        
        # Load test data
        features, labels = classifier.load_real_training_data(data_file)
        
        # Make predictions
        features_scaled = classifier.scaler.transform(features)
        predictions = classifier.model.predict(features_scaled)
        probabilities = classifier.model.predict_proba(features_scaled)
        
        # Classification report
        gesture_names = list(classifier.gesture_labels.values())
        print("\nClassification Report:")
        print(classification_report(labels, predictions, target_names=gesture_names))
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=gesture_names, yticklabels=gesture_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'confusion_matrix.png'))
        plt.show()
        
        # Feature importance (if available)
        if hasattr(classifier.model, 'feature_importances_'):
            self._plot_feature_importance(classifier.model.feature_importances_)
    
    def _plot_feature_importance(self, feature_importances: np.ndarray):
        """Plot feature importance for tree-based models."""
        # Create feature names (simplified)
        feature_names = []
        feature_names.extend([f'coord_{i}' for i in range(15)])  # Coordinate features
        feature_names.extend([f'dist_{i}' for i in range(19)])   # Distance features  
        feature_names.extend([f'angle_{i}' for i in range(18)])  # Angle features
        feature_names.extend([f'state_{i}' for i in range(15)])  # Finger state features
        feature_names.extend([f'shape_{i}' for i in range(5)])   # Shape features
        feature_names.extend([f'geom_{i}' for i in range(8)])    # Geometric features
        
        # Only use available feature names
        n_features = len(feature_importances)
        feature_names = feature_names[:n_features]
        
        # Get top 20 most important features
        indices = np.argsort(feature_importances)[::-1][:20]
        
        plt.figure(figsize=(12, 8))
        plt.title('Top 20 Feature Importances')
        plt.bar(range(20), feature_importances[indices])
        plt.xticks(range(20), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'feature_importance.png'))
        plt.show()

def main():
    """Main training pipeline."""
    trainer = ImprovedGestureTrainer()
    
    print("Improved Gesture Recognition Training Pipeline")
    print("=" * 50)
    
    # Step 1: Collect training data (optional - skip if data exists)
    data_file = os.path.join(trainer.data_dir, "training_data.json")
    
    if not os.path.exists(data_file):
        print("No training data found. Starting data collection...")
        data_file = trainer.collect_training_data()
    else:
        print(f"Using existing training data: {data_file}")
    
    # Step 2: Train and compare multiple models
    best_model_name, best_model = trainer.train_multiple_models(data_file)
    
    # Step 3: Hyperparameter tuning
    optimized_model, best_params = trainer.hyperparameter_tuning(data_file, best_model_name)
    
    # Step 4: Create final optimized classifier
    final_classifier = trainer.create_optimized_classifier(data_file, best_params, best_model_name)
    
    # Step 5: Evaluate the final model
    trainer.evaluate_model(final_classifier, data_file)
    
    # Step 6: Save the trained model
    model_path = os.path.join(trainer.data_dir, "optimized_gesture_model.pkl")
    final_classifier.save_model(model_path)
    
    print(f"\nTraining complete!")
    print(f"Optimized model saved to: {model_path}")
    print(f"Best model: {best_model_name}")
    print(f"Best parameters: {best_params}")
    
    return final_classifier

if __name__ == "__main__":
    main() 