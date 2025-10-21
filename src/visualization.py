"""
Visualization utilities for action recognition results.

This module provides functions for creating visualizations of action recognition
results, including confidence plots, frame analysis, and result summaries.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple, Optional, Union
from pathlib import Path
import cv2
import logging

logger = logging.getLogger(__name__)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_prediction_confidence(
    predictions: List[Tuple[str, float]],
    title: str = "Action Recognition Results",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Create a horizontal bar plot showing prediction confidence scores.
    
    Args:
        predictions: List of (action_name, confidence) tuples
        title: Plot title
        save_path: Optional path to save the plot
        figsize: Figure size (width, height)
    """
    if not predictions:
        logger.warning("No predictions to plot")
        return
    
    # Extract data
    actions = [pred[0] for pred in predictions]
    confidences = [pred[1] for pred in predictions]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    y_pos = np.arange(len(actions))
    bars = ax.barh(y_pos, confidences, alpha=0.8)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(actions)
    ax.set_xlabel('Confidence Score')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    
    # Add confidence values on bars
    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{conf:.3f}', ha='left', va='center')
    
    # Color bars by confidence (darker = higher confidence)
    for bar, conf in zip(bars, confidences):
        bar.set_color(plt.cm.viridis(conf))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confidence plot saved to: {save_path}")
    
    plt.show()


def create_prediction_summary(
    predictions: List[Tuple[str, float]],
    video_path: Union[str, Path],
    model_name: str,
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Create a comprehensive summary visualization of prediction results.
    
    Args:
        predictions: List of (action_name, confidence) tuples
        video_path: Path to the analyzed video
        model_name: Name of the model used
        save_path: Optional path to save the summary
    """
    if not predictions:
        logger.warning("No predictions to summarize")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Main title
    fig.suptitle(f'Action Recognition Analysis Summary\n'
                 f'Video: {Path(video_path).name} | Model: {model_name}', 
                 fontsize=16, fontweight='bold')
    
    # Top prediction (large display)
    ax1 = plt.subplot(2, 2, 1)
    top_action, top_confidence = predictions[0]
    
    # Create a large confidence circle
    circle = plt.Circle((0.5, 0.5), 0.3, color=plt.cm.viridis(top_confidence), alpha=0.7)
    ax1.add_patch(circle)
    ax1.text(0.5, 0.5, f'{top_action}\n{top_confidence:.1%}', 
             ha='center', va='center', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('Top Prediction', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Confidence bar chart
    ax2 = plt.subplot(2, 2, 2)
    actions = [pred[0] for pred in predictions]
    confidences = [pred[1] for pred in predictions]
    
    bars = ax2.bar(range(len(actions)), confidences, alpha=0.8)
    ax2.set_xticks(range(len(actions)))
    ax2.set_xticklabels(actions, rotation=45, ha='right')
    ax2.set_ylabel('Confidence Score')
    ax2.set_title('All Predictions', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    
    # Color bars
    for bar, conf in zip(bars, confidences):
        bar.set_color(plt.cm.viridis(conf))
    
    # Confidence distribution
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_xlabel('Confidence Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax3.axvline(np.mean(confidences), color='red', linestyle='--', 
                label=f'Mean: {np.mean(confidences):.3f}')
    ax3.legend()
    
    # Model and video info
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    info_text = f"""
    Model: {model_name}
    Video: {Path(video_path).name}
    Total Predictions: {len(predictions)}
    Top Confidence: {top_confidence:.3f}
    Mean Confidence: {np.mean(confidences):.3f}
    Std Confidence: {np.std(confidences):.3f}
    """
    
    ax4.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Summary plot saved to: {save_path}")
    
    plt.show()


def visualize_video_frames(
    video_path: Union[str, Path],
    frame_indices: Optional[List[int]] = None,
    max_frames: int = 8,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Visualize key frames from a video.
    
    Args:
        video_path: Path to the video file
        frame_indices: Specific frame indices to extract
        max_frames: Maximum number of frames to display
        save_path: Optional path to save the visualization
        figsize: Figure size
    """
    import cv2
    
    # Extract frames
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
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    
    if not frames:
        logger.warning("No frames extracted from video")
        return
    
    # Create subplot grid
    n_frames = len(frames)
    cols = min(4, n_frames)
    rows = (n_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, frame in enumerate(frames):
        if i < len(axes):
            axes[i].imshow(frame)
            axes[i].set_title(f'Frame {frame_indices[i]}')
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(frames), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Key Frames from {Path(video_path).name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Frame visualization saved to: {save_path}")
    
    plt.show()


def create_comparison_plot(
    results_dict: dict,
    title: str = "Model Comparison",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Create a comparison plot for multiple model results.
    
    Args:
        results_dict: Dictionary with model names as keys and predictions as values
        title: Plot title
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    if not results_dict:
        logger.warning("No results to compare")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Extract all unique actions
    all_actions = set()
    for predictions in results_dict.values():
        all_actions.update([pred[0] for pred in predictions])
    all_actions = sorted(list(all_actions))
    
    # Create comparison matrix
    comparison_data = []
    for model_name, predictions in results_dict.items():
        model_scores = []
        pred_dict = {pred[0]: pred[1] for pred in predictions}
        for action in all_actions:
            model_scores.append(pred_dict.get(action, 0.0))
        comparison_data.append(model_scores)
    
    comparison_data = np.array(comparison_data)
    
    # Heatmap
    ax1 = axes[0, 0]
    sns.heatmap(comparison_data, 
                xticklabels=all_actions,
                yticklabels=list(results_dict.keys()),
                annot=True, fmt='.3f', cmap='viridis', ax=ax1)
    ax1.set_title('Confidence Scores Heatmap')
    ax1.set_xlabel('Actions')
    ax1.set_ylabel('Models')
    
    # Top predictions per model
    ax2 = axes[0, 1]
    for i, (model_name, predictions) in enumerate(results_dict.items()):
        top_action, top_confidence = predictions[0]
        ax2.bar(i, top_confidence, label=f'{model_name}: {top_action}')
    ax2.set_title('Top Prediction Confidence')
    ax2.set_ylabel('Confidence Score')
    ax2.set_xticks(range(len(results_dict)))
    ax2.set_xticklabels(list(results_dict.keys()))
    ax2.legend()
    
    # Confidence distribution
    ax3 = axes[1, 0]
    for model_name, predictions in results_dict.items():
        confidences = [pred[1] for pred in predictions]
        ax3.hist(confidences, alpha=0.6, label=model_name, bins=10)
    ax3.set_title('Confidence Distribution')
    ax3.set_xlabel('Confidence Score')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # Model performance summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "Model Performance Summary:\n\n"
    for model_name, predictions in results_dict.items():
        confidences = [pred[1] for pred in predictions]
        summary_text += f"{model_name}:\n"
        summary_text += f"  Mean Confidence: {np.mean(confidences):.3f}\n"
        summary_text += f"  Std Confidence: {np.std(confidences):.3f}\n"
        summary_text += f"  Top Action: {predictions[0][0]}\n\n"
    
    ax4.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to: {save_path}")
    
    plt.show()


def main():
    """Example usage of visualization functions."""
    # Example predictions
    sample_predictions = [
        ("walking", 0.85),
        ("running", 0.12),
        ("jumping", 0.02),
        ("sitting", 0.01)
    ]
    
    # Plot confidence
    plot_prediction_confidence(sample_predictions, "Sample Results")
    
    # Create summary
    create_prediction_summary(sample_predictions, "sample_video.mp4", "r3d_18")


if __name__ == "__main__":
    main()
