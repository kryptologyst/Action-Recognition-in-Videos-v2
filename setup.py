#!/usr/bin/env python3
"""
Setup script for Action Recognition in Videos project.

This script helps set up the project environment and dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def create_directories():
    """Create necessary directories."""
    directories = ["data", "models", "logs", "config"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")


def install_dependencies():
    """Install project dependencies."""
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("💡 Try running: pip install --upgrade pip")
        return False
    return True


def verify_installation():
    """Verify that key packages are installed."""
    packages = ["torch", "torchvision", "streamlit", "opencv-python", "numpy"]
    
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is not available")
            return False
    return True


def run_tests():
    """Run the test suite."""
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Some tests failed, but the basic setup should still work")
        return False
    return True


def main():
    """Main setup function."""
    print("🚀 Setting up Action Recognition in Videos")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating project directories...")
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        print("Please check your Python environment and try again")
        sys.exit(1)
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    if not verify_installation():
        print("❌ Some required packages are missing")
        sys.exit(1)
    
    # Run tests
    print("\n🧪 Running tests...")
    run_tests()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the example: python example.py")
    print("2. Start the web app: streamlit run web_app/app.py")
    print("3. Use the CLI: python src/cli.py predict your_video.mp4")
    print("4. Read the README.md for more information")


if __name__ == "__main__":
    main()
