#!/usr/bin/env python3
"""
Quick Structure Test
===================

Test script to verify the new organized hand tracking structure works correctly.
"""

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        # Test main package imports
        from . import GestureDetector, GestureClassifier, GestureDataCollector
        print("✓ Main package imports work")
        
        # Test core module imports
        from .core import GestureDetector as CoreGD, GestureClassifier as CoreGC
        print("✓ Core module imports work")
        
        # Test data module imports
        from .data import GestureDataCollector as DataGDC
        print("✓ Data module imports work")
        
        # Test training module imports
        from .training import ImprovedGestureTrainer
        print("✓ Training module imports work")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of core classes."""
    print("\nTesting basic functionality...")
    
    try:
        from .core import GestureDetector, GestureClassifier
        
        # Test GestureDetector initialization
        detector = GestureDetector()
        print("✓ GestureDetector creation successful")
        
        # Test GestureClassifier initialization
        classifier = GestureClassifier()
        print("✓ GestureClassifier creation successful")
        
        # Test gesture info
        info = classifier.get_gesture_info()
        print(f"✓ Gesture info: {len(info['supported_gestures'])} gestures supported")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality error: {e}")
        return False

def test_structure():
    """Test directory structure."""
    print("\nTesting directory structure...")
    
    import os
    import os.path as path
    
    # Check required directories
    required_dirs = ['core', 'data', 'training', 'demos', 'tests']
    for dir_name in required_dirs:
        if path.exists(dir_name) and path.isdir(dir_name):
            print(f"✓ Directory '{dir_name}' exists")
        else:
            print(f"✗ Directory '{dir_name}' missing")
            return False
    
    # Check required files
    required_files = [
        'core/__init__.py',
        'core/gesture_detector.py', 
        'core/gesture_classifier.py',
        'data/__init__.py',
        'data/data_collector.py',
        'training/__init__.py',
        'training/train_improved_model.py',
        'demos/__init__.py',
        'tests/__init__.py'
    ]
    
    for file_path in required_files:
        if path.exists(file_path) and path.isfile(file_path):
            print(f"✓ File '{file_path}' exists")
        else:
            print(f"✗ File '{file_path}' missing")
            return False
    
    return True

def main():
    """Run all tests."""
    print("Hand Tracking Structure Test")
    print("=" * 40)
    
    # Run tests
    tests = [
        test_structure,
        test_imports,
        test_basic_functionality
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"Test {test.__name__} failed!")
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! New structure is working correctly.")
    else:
        print("❌ Some tests failed. Check the structure and imports.")

if __name__ == "__main__":
    main() 