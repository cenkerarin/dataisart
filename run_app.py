#!/usr/bin/env python3
"""
Application Launcher
===================

Launcher script for the Hands-Free Data Science PyQt desktop application.
Checks dependencies and provides helpful error messages.
"""

import sys
import subprocess

def check_pyqt5():
    """Check if PyQt5 is available."""
    try:
        import PyQt5
        print("✅ PyQt5 is available")
        return True
    except ImportError:
        print("❌ PyQt5 is not installed")
        print()
        print("To install PyQt5, run:")
        print("  pip install PyQt5")
        print("  # or")
        print("  pip install -r requirements.txt")
        print()
        return False

def check_opencv():
    """Check if OpenCV is available."""
    try:
        import cv2
        print("✅ OpenCV is available")
        return True
    except ImportError:
        print("❌ OpenCV is not installed")
        print()
        print("To install OpenCV, run:")
        print("  pip install opencv-python")
        print()
        return False

def main():
    """Main launcher function."""
    print("🧠🖐️ Hands-Free Data Science Application Launcher")
    print("=" * 55)
    print()
    
    print("Checking dependencies...")
    
    # Check required dependencies
    pyqt5_ok = check_pyqt5()
    opencv_ok = check_opencv()
    
    if not (pyqt5_ok and opencv_ok):
        print()
        print("❌ Missing required dependencies. Please install them first.")
        print()
        print("Quick install command:")
        print("  pip install PyQt5 opencv-python")
        return 1
    
    print()
    print("✅ All dependencies available!")
    print("🚀 Starting application...")
    print()
    
    # Import and run the main application
    try:
        from main import main as app_main
        return app_main()
    except Exception as e:
        print(f"❌ Error starting application: {str(e)}")
        print()
        print("Troubleshooting tips:")
        print("1. Make sure you're in the project root directory")
        print("2. Check that all dependencies are installed")
        print("3. Try running: python main.py")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 