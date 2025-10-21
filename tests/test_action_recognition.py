"""
Unit tests for the action recognition module.

This module contains comprehensive tests for the ActionRecognitionModel
and related utilities.
"""

import unittest
import tempfile
import os
from pathlib import Path
import torch
import numpy as np

# Add src directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from action_recognition import (
    ActionRecognitionModel, 
    create_synthetic_video, 
    VideoAnalyzer
)


class TestActionRecognitionModel(unittest.TestCase):
    """Test cases for ActionRecognitionModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_video_path = Path(self.temp_dir) / "test_video.mp4"
        
        # Create a synthetic test video
        create_synthetic_video(self.test_video_path, "walking", 2.0)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = ActionRecognitionModel(model_name="r3d_18")
        
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.classes)
        self.assertIsNotNone(model.transform)
        self.assertEqual(model.model_name, "r3d_18")
    
    def test_preprocess_video(self):
        """Test video preprocessing."""
        model = ActionRecognitionModel(model_name="r3d_18")
        
        video_tensor = model.preprocess_video(self.test_video_path)
        
        # Check tensor shape: (1, C, T, H, W)
        self.assertEqual(len(video_tensor.shape), 5)
        self.assertEqual(video_tensor.shape[0], 1)  # batch size
        self.assertEqual(video_tensor.shape[1], 3)   # channels
        self.assertEqual(video_tensor.shape[2], 32)  # time frames
        self.assertEqual(video_tensor.shape[3], 112) # height
        self.assertEqual(video_tensor.shape[4], 112) # width
    
    def test_predict(self):
        """Test prediction functionality."""
        model = ActionRecognitionModel(model_name="r3d_18")
        
        video_tensor = model.preprocess_video(self.test_video_path)
        predictions = model.predict(video_tensor, top_k=3)
        
        # Check predictions format
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 3)
        
        for action, confidence in predictions:
            self.assertIsInstance(action, str)
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
    
    def test_predict_from_file(self):
        """Test prediction from file."""
        model = ActionRecognitionModel(model_name="r3d_18")
        
        predictions = model.predict_from_file(self.test_video_path, top_k=5)
        
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 5)
        
        # Check that predictions are sorted by confidence
        confidences = [conf for _, conf in predictions]
        self.assertEqual(confidences, sorted(confidences, reverse=True))
    
    def test_invalid_model_name(self):
        """Test handling of invalid model name."""
        with self.assertRaises(ValueError):
            ActionRecognitionModel(model_name="invalid_model")


class TestVideoAnalyzer(unittest.TestCase):
    """Test cases for VideoAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_video_path = Path(self.temp_dir) / "test_video.mp4"
        
        # Create a synthetic test video
        create_synthetic_video(self.test_video_path, "jumping", 3.0)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_extract_frames(self):
        """Test frame extraction."""
        frames = VideoAnalyzer.extract_frames(self.test_video_path, max_frames=8)
        
        self.assertIsInstance(frames, list)
        self.assertGreater(len(frames), 0)
        self.assertLessEqual(len(frames), 8)
        
        # Check frame format
        for frame in frames:
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(len(frame.shape), 3)  # H, W, C
    
    def test_extract_specific_frames(self):
        """Test extraction of specific frame indices."""
        frame_indices = [0, 5, 10, 15]
        frames = VideoAnalyzer.extract_frames(self.test_video_path, frame_indices=frame_indices)
        
        self.assertIsInstance(frames, list)
        self.assertLessEqual(len(frames), len(frame_indices))
    
    def test_create_video_summary(self):
        """Test video summary creation."""
        output_path = Path(self.temp_dir) / "summary.mp4"
        
        VideoAnalyzer.create_video_summary(self.test_video_path, output_path, max_frames=4)
        
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)


class TestSyntheticVideo(unittest.TestCase):
    """Test cases for synthetic video creation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_synthetic_video(self):
        """Test synthetic video creation."""
        output_path = Path(self.temp_dir) / "synthetic.mp4"
        
        create_synthetic_video(output_path, "walking", 2.0, 30)
        
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)
    
    def test_create_synthetic_video_different_actions(self):
        """Test creating synthetic videos with different actions."""
        actions = ["walking", "jumping", "static"]
        
        for action in actions:
            output_path = Path(self.temp_dir) / f"synthetic_{action}.mp4"
            create_synthetic_video(output_path, action, 1.0, 30)
            self.assertTrue(output_path.exists())


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_video_path = Path(self.temp_dir) / "integration_test.mp4"
        
        # Create a synthetic test video
        create_synthetic_video(self.test_video_path, "walking", 3.0)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_prediction(self):
        """Test complete end-to-end prediction pipeline."""
        model = ActionRecognitionModel(model_name="r3d_18")
        
        # Extract frames
        frames = VideoAnalyzer.extract_frames(self.test_video_path, max_frames=16)
        self.assertGreater(len(frames), 0)
        
        # Make prediction
        predictions = model.predict_from_file(self.test_video_path, top_k=3)
        self.assertEqual(len(predictions), 3)
        
        # Verify prediction format
        for action, confidence in predictions:
            self.assertIsInstance(action, str)
            self.assertIsInstance(confidence, float)
            self.assertGreater(len(action), 0)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
