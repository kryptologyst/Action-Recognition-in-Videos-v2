#!/usr/bin/env python3
"""
Example script demonstrating the Action Recognition system.

This script shows how to use the action recognition system programmatically
and demonstrates various features and capabilities.
"""

import sys
from pathlib import Path
import logging

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from action_recognition import ActionRecognitionModel, create_synthetic_video, VideoAnalyzer
from visualization import plot_prediction_confidence, create_prediction_summary
from config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_basic_usage():
    """Demonstrate basic usage of the action recognition system."""
    print("🎬 Action Recognition Demo")
    print("=" * 50)
    
    # Create a synthetic test video
    test_video_path = "data/demo_video.mp4"
    print(f"Creating synthetic test video: {test_video_path}")
    create_synthetic_video(test_video_path, "walking", 3.0)
    
    # Initialize model
    print("Loading R3D-18 model...")
    model = ActionRecognitionModel(model_name="r3d_18")
    
    # Make prediction
    print("Analyzing video...")
    predictions = model.predict_from_file(test_video_path, top_k=5)
    
    # Display results
    print("\n🎯 Prediction Results:")
    print("-" * 30)
    for i, (action, confidence) in enumerate(predictions, 1):
        print(f"{i:2d}. {action:<20} {confidence:.3f} ({confidence:.1%})")
    
    return predictions, test_video_path


def demonstrate_model_comparison():
    """Demonstrate comparison between different model architectures."""
    print("\n🔍 Model Comparison Demo")
    print("=" * 50)
    
    # Create test video
    test_video_path = "data/comparison_video.mp4"
    create_synthetic_video(test_video_path, "jumping", 4.0)
    
    # Test different models
    models = ["r3d_18", "mc3_18", "r2plus1d_18"]
    results = {}
    
    for model_name in models:
        print(f"\nTesting {model_name}...")
        try:
            model = ActionRecognitionModel(model_name=model_name)
            predictions = model.predict_from_file(test_video_path, top_k=3)
            results[model_name] = predictions
            
            print(f"Top prediction: {predictions[0][0]} ({predictions[0][1]:.3f})")
            
        except Exception as e:
            logger.error(f"Failed to test {model_name}: {e}")
            continue
    
    return results, test_video_path


def demonstrate_video_analysis():
    """Demonstrate advanced video analysis features."""
    print("\n📊 Video Analysis Demo")
    print("=" * 50)
    
    # Create test video
    test_video_path = "data/analysis_video.mp4"
    create_synthetic_video(test_video_path, "static", 5.0)
    
    # Extract frames
    print("Extracting key frames...")
    frames = VideoAnalyzer.extract_frames(test_video_path, max_frames=8)
    print(f"Extracted {len(frames)} frames")
    
    # Create video summary
    summary_path = "data/video_summary.mp4"
    print(f"Creating video summary: {summary_path}")
    VideoAnalyzer.create_video_summary(test_video_path, summary_path, max_frames=4)
    
    return test_video_path, summary_path


def demonstrate_visualization():
    """Demonstrate visualization capabilities."""
    print("\n📈 Visualization Demo")
    print("=" * 50)
    
    # Get some sample predictions
    predictions, video_path = demonstrate_basic_usage()
    
    # Create confidence plot
    print("Creating confidence visualization...")
    plot_prediction_confidence(predictions, "Demo Results")
    
    # Create comprehensive summary
    print("Creating prediction summary...")
    create_prediction_summary(predictions, video_path, "r3d_18")


def demonstrate_configuration():
    """Demonstrate configuration management."""
    print("\n⚙️ Configuration Demo")
    print("=" * 50)
    
    # Load configuration
    config = get_config()
    
    print("Current Configuration:")
    print(f"Model: {config.model.name}")
    print(f"Device: {config.model.device or 'Auto-detect'}")
    print(f"Max Frames: {config.model.max_frames}")
    print(f"Data Directory: {config.data.data_dir}")
    print(f"Logging Level: {config.logging.level}")


def main():
    """Main demonstration function."""
    print("🚀 Action Recognition System - Complete Demo")
    print("=" * 60)
    
    try:
        # Basic usage
        demonstrate_basic_usage()
        
        # Model comparison
        demonstrate_model_comparison()
        
        # Video analysis
        demonstrate_video_analysis()
        
        # Configuration
        demonstrate_configuration()
        
        # Visualization (optional - requires matplotlib)
        try:
            demonstrate_visualization()
        except ImportError:
            print("\n📈 Visualization Demo Skipped (matplotlib not available)")
        
        print("\n✅ Demo completed successfully!")
        print("\nNext steps:")
        print("1. Run the web interface: streamlit run web_app/app.py")
        print("2. Use CLI: python src/cli.py predict your_video.mp4")
        print("3. Check the README.md for more examples")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        print("Please check the error messages above and ensure all dependencies are installed.")


if __name__ == "__main__":
    main()
