"""
Action Recognition in Videos

This module provides a modern implementation of action recognition using 3D CNNs.
It supports multiple model architectures and provides utilities for video preprocessing,
inference, and visualization.

Author: AI Assistant
Date: 2024
"""

import logging
import json
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
import warnings

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models.video as video_models
import cv2
import numpy as np
from torchvision.io import read_video
from PIL import Image
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionRecognitionModel:
    """
    A modern action recognition model wrapper supporting multiple architectures.
    
    This class provides a unified interface for loading and using various
    3D CNN models for action recognition tasks.
    """
    
    def __init__(
        self,
        model_name: str = "r3d_18",
        device: Optional[str] = None,
        num_classes: int = 400
    ):
        """
        Initialize the action recognition model.
        
        Args:
            model_name: Name of the model architecture ('r3d_18', 'mc3_18', 'r2plus1d_18')
            device: Device to run the model on ('cpu', 'cuda', or None for auto-detection)
            num_classes: Number of action classes (default: 400 for Kinetics-400)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.model: Optional[nn.Module] = None
        self.classes: List[str] = []
        self.transform = None
        
        self._load_model()
        self._load_classes()
        self._setup_transforms()
        
        logger.info(f"Action recognition model '{model_name}' loaded on {self.device}")
    
    def _load_model(self) -> None:
        """Load the specified model architecture."""
        try:
            if self.model_name == "r3d_18":
                # Use weights parameter instead of deprecated pretrained
                self.model = video_models.r3d_18(weights=video_models.R3D_18_Weights.KINETICS400_V1)
            elif self.model_name == "mc3_18":
                self.model = video_models.mc3_18(weights=video_models.MC3_18_Weights.KINETICS400_V1)
            elif self.model_name == "r2plus1d_18":
                self.model = video_models.r2plus1d_18(weights=video_models.R2Plus1D_18_Weights.KINETICS400_V1)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def _load_classes(self) -> None:
        """Load the class labels for Kinetics-400 dataset."""
        try:
            # Try to load from local file first
            local_labels_path = Path("data/kinetics_labels.txt")
            if local_labels_path.exists():
                with open(local_labels_path, 'r') as f:
                    self.classes = [line.strip() for line in f.readlines()]
                logger.info(f"Loaded {len(self.classes)} classes from local file")
                return
            
            # Fallback to downloading from GitHub
            url = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
            with urllib.request.urlopen(url) as f:
                self.classes = [line.decode('utf-8').strip() for line in f.readlines()]
            
            # Save locally for future use
            local_labels_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_labels_path, 'w') as f:
                f.write('\n'.join(self.classes))
            
            logger.info(f"Downloaded and cached {len(self.classes)} classes")
            
        except Exception as e:
            logger.warning(f"Failed to load classes: {e}")
            # Create fallback classes
            self.classes = [f"action_{i}" for i in range(self.num_classes)]
            logger.info(f"Using fallback classes: {len(self.classes)}")
    
    def _setup_transforms(self) -> None:
        """Setup video preprocessing transforms."""
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645],
                std=[0.22803, 0.22145, 0.216989]
            )
        ])
    
    def preprocess_video(
        self,
        video_path: Union[str, Path],
        max_frames: int = 32
    ) -> torch.Tensor:
        """
        Preprocess a video file for model inference.
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to process
            
        Returns:
            Preprocessed video tensor of shape (1, C, T, H, W)
        """
        try:
            # Read video
            video, _, _ = read_video(str(video_path), pts_unit='sec')
            
            # Take first max_frames frames
            video = video[:max_frames]
            
            # Convert to float and normalize
            video = video.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
            
            # Apply transforms
            video = self.transform(video)
            
            # Add batch dimension and rearrange to (B, C, T, H, W)
            video = video.unsqueeze(0).permute(0, 2, 1, 3, 4)
            
            return video.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to preprocess video {video_path}: {e}")
            raise
    
    def predict(
        self,
        video_tensor: torch.Tensor,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Predict actions from preprocessed video tensor.
        
        Args:
            video_tensor: Preprocessed video tensor
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_name, confidence) tuples
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        with torch.no_grad():
            outputs = self.model(video_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
            
            results = []
            for i in range(top_k):
                class_idx = top_indices[0, i].item()
                confidence = top_probs[0, i].item()
                class_name = self.classes[class_idx] if class_idx < len(self.classes) else f"unknown_{class_idx}"
                results.append((class_name, confidence))
            
            return results
    
    def predict_from_file(
        self,
        video_path: Union[str, Path],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Predict actions directly from a video file.
        
        Args:
            video_path: Path to the video file
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_name, confidence) tuples
        """
        video_tensor = self.preprocess_video(video_path)
        return self.predict(video_tensor, top_k)


class VideoAnalyzer:
    """
    Advanced video analysis utilities for action recognition.
    """
    
    @staticmethod
    def extract_frames(
        video_path: Union[str, Path],
        frame_indices: Optional[List[int]] = None,
        max_frames: int = 32
    ) -> List[np.ndarray]:
        """
        Extract specific frames from a video.
        
        Args:
            video_path: Path to the video file
            frame_indices: Specific frame indices to extract (None for uniform sampling)
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of frame arrays
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        if frame_indices is None:
            # Uniform sampling
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    @staticmethod
    def create_video_summary(
        video_path: Union[str, Path],
        output_path: Union[str, Path],
        max_frames: int = 16
    ) -> None:
        """
        Create a summary video with key frames.
        
        Args:
            video_path: Input video path
            output_path: Output summary video path
            max_frames: Number of frames in summary
        """
        frames = VideoAnalyzer.extract_frames(video_path, max_frames=max_frames)
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        # Get video properties
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Create output video
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        logger.info(f"Video summary created: {output_path}")


def create_synthetic_video(
    output_path: Union[str, Path],
    action_type: str = "walking",
    duration: float = 3.0,
    fps: int = 30
) -> None:
    """
    Create a synthetic video for testing purposes.
    
    Args:
        output_path: Path to save the synthetic video
        action_type: Type of action to simulate
        duration: Video duration in seconds
        fps: Frames per second
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a simple synthetic video
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    total_frames = int(duration * fps)
    
    for frame_idx in range(total_frames):
        # Create a simple moving pattern based on action type
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        if action_type == "walking":
            # Simulate walking with moving rectangles
            x = int((frame_idx / total_frames) * (width - 100))
            cv2.rectangle(frame, (x, height//2), (x + 50, height//2 + 100), (255, 255, 255), -1)
        elif action_type == "jumping":
            # Simulate jumping with vertical movement
            y = int(height//2 + 50 * np.sin(frame_idx * 0.3))
            cv2.circle(frame, (width//2, y), 30, (255, 255, 255), -1)
        else:
            # Default: static object
            cv2.circle(frame, (width//2, height//2), 30, (255, 255, 255), -1)
        
        out.write(frame)
    
    out.release()
    logger.info(f"Synthetic video created: {output_path}")


def main():
    """Example usage of the action recognition system."""
    # Create synthetic test video
    test_video_path = "data/test_video.mp4"
    create_synthetic_video(test_video_path, action_type="walking")
    
    # Initialize model
    model = ActionRecognitionModel(model_name="r3d_18")
    
    # Make prediction
    try:
        predictions = model.predict_from_file(test_video_path, top_k=5)
        
        print("\n🎬 Action Recognition Results:")
        print("=" * 50)
        for i, (action, confidence) in enumerate(predictions, 1):
            print(f"{i}. {action:<20} {confidence:.3f}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
