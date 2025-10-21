"""
Command Line Interface for Action Recognition in Videos.

This module provides a CLI for running action recognition on video files
from the command line.
"""

import argparse
import sys
from pathlib import Path
import logging
from typing import Optional

from action_recognition import ActionRecognitionModel, create_synthetic_video, VideoAnalyzer
from config import get_config


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def predict_action(
    video_path: str,
    model_name: str = "r3d_18",
    top_k: int = 5,
    output_file: Optional[str] = None
) -> None:
    """
    Predict actions in a video file.
    
    Args:
        video_path: Path to the video file
        model_name: Model architecture to use
        top_k: Number of top predictions to show
        output_file: Optional file to save results
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load model
        logger.info(f"Loading {model_name} model...")
        model = ActionRecognitionModel(model_name=model_name)
        
        # Make prediction
        logger.info(f"Analyzing video: {video_path}")
        predictions = model.predict_from_file(video_path, top_k=top_k)
        
        # Display results
        print(f"\n🎬 Action Recognition Results for: {video_path}")
        print("=" * 60)
        
        results_text = []
        for i, (action, confidence) in enumerate(predictions, 1):
            result_line = f"{i:2d}. {action:<25} {confidence:.3f} ({confidence:.1%})"
            print(result_line)
            results_text.append(result_line)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"Action Recognition Results for: {video_path}\n")
                f.write("=" * 60 + "\n")
                f.write("\n".join(results_text))
            logger.info(f"Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


def create_test_video(
    output_path: str,
    action_type: str = "walking",
    duration: float = 3.0,
    fps: int = 30
) -> None:
    """
    Create a synthetic test video.
    
    Args:
        output_path: Path to save the video
        action_type: Type of action to simulate
        duration: Video duration in seconds
        fps: Frames per second
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Creating synthetic video: {output_path}")
        create_synthetic_video(output_path, action_type, duration, fps)
        print(f"✅ Synthetic video created: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create synthetic video: {e}")
        sys.exit(1)


def batch_predict(
    input_dir: str,
    model_name: str = "r3d_18",
    output_dir: Optional[str] = None,
    top_k: int = 5
) -> None:
    """
    Run action recognition on all videos in a directory.
    
    Args:
        input_dir: Directory containing video files
        model_name: Model architecture to use
        output_dir: Directory to save results (optional)
        top_k: Number of top predictions per video
    """
    logger = logging.getLogger(__name__)
    
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Find video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    video_files = [
        f for f in input_path.iterdir() 
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    
    if not video_files:
        logger.error(f"No video files found in: {input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(video_files)} video files")
    
    # Load model once
    model = ActionRecognitionModel(model_name=model_name)
    
    # Process each video
    for video_file in video_files:
        try:
            logger.info(f"Processing: {video_file.name}")
            predictions = model.predict_from_file(video_file, top_k=top_k)
            
            # Display results
            print(f"\n📹 {video_file.name}")
            print("-" * 40)
            for i, (action, confidence) in enumerate(predictions, 1):
                print(f"{i:2d}. {action:<20} {confidence:.3f}")
            
            # Save to file if output directory specified
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                result_file = output_path / f"{video_file.stem}_results.txt"
                with open(result_file, 'w') as f:
                    f.write(f"Action Recognition Results for: {video_file.name}\n")
                    f.write("-" * 40 + "\n")
                    for i, (action, confidence) in enumerate(predictions, 1):
                        f.write(f"{i:2d}. {action:<20} {confidence:.3f}\n")
                
                logger.info(f"Results saved to: {result_file}")
        
        except Exception as e:
            logger.error(f"Failed to process {video_file.name}: {e}")
            continue


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Action Recognition in Videos - CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single video
  python cli.py predict video.mp4
  
  # Use different model
  python cli.py predict video.mp4 --model mc3_18
  
  # Create test video
  python cli.py create-test --output test.mp4 --action jumping
  
  # Batch process directory
  python cli.py batch videos/ --output-dir results/
        """
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict actions in a video')
    predict_parser.add_argument('video_path', help='Path to video file')
    predict_parser.add_argument(
        '--model',
        choices=['r3d_18', 'mc3_18', 'r2plus1d_18'],
        default='r3d_18',
        help='Model architecture to use'
    )
    predict_parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top predictions to show'
    )
    predict_parser.add_argument(
        '--output',
        help='File to save results'
    )
    
    # Create test video command
    create_parser = subparsers.add_parser('create-test', help='Create synthetic test video')
    create_parser.add_argument(
        '--output',
        required=True,
        help='Output video file path'
    )
    create_parser.add_argument(
        '--action',
        choices=['walking', 'jumping', 'static'],
        default='walking',
        help='Type of action to simulate'
    )
    create_parser.add_argument(
        '--duration',
        type=float,
        default=3.0,
        help='Video duration in seconds'
    )
    create_parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second'
    )
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process multiple videos')
    batch_parser.add_argument('input_dir', help='Directory containing video files')
    batch_parser.add_argument(
        '--model',
        choices=['r3d_18', 'mc3_18', 'r2plus1d_18'],
        default='r3d_18',
        help='Model architecture to use'
    )
    batch_parser.add_argument(
        '--output-dir',
        help='Directory to save results'
    )
    batch_parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top predictions per video'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    if args.command == 'predict':
        predict_action(
            args.video_path,
            args.model,
            args.top_k,
            args.output
        )
    elif args.command == 'create-test':
        create_test_video(
            args.output,
            args.action,
            args.duration,
            args.fps
        )
    elif args.command == 'batch':
        batch_predict(
            args.input_dir,
            args.model,
            args.output_dir,
            args.top_k
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
