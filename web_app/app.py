"""
Streamlit web application for Action Recognition in Videos.

This module provides a user-friendly web interface for uploading videos
and getting action recognition predictions.
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import sys
import logging

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from action_recognition import ActionRecognitionModel, create_synthetic_video, VideoAnalyzer
from config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Action Recognition in Videos",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-bar {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background-color: #1f77b4;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name: str = "r3d_18"):
    """Load and cache the action recognition model."""
    try:
        return ActionRecognitionModel(model_name=model_name)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def display_prediction_results(predictions, model_name):
    """Display prediction results in a formatted way."""
    st.markdown("### 🎯 Prediction Results")
    
    # Display top prediction prominently
    if predictions:
        top_action, top_confidence = predictions[0]
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Most Likely Action:** {top_action}")
        with col2:
            st.metric("Confidence", f"{top_confidence:.1%}")
        
        # Confidence bar
        st.markdown('<div class="confidence-bar">', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence-fill" style="width: {top_confidence*100}%"></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # All predictions
        st.markdown("#### All Predictions:")
        for i, (action, confidence) in enumerate(predictions, 1):
            with st.container():
                st.markdown(f"""
                <div class="prediction-card">
                    <strong>{i}. {action}</strong><br>
                    Confidence: {confidence:.3f} ({confidence:.1%})
                </div>
                """, unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">🎬 Action Recognition in Videos</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Model selection
    model_options = {
        "R3D-18": "r3d_18",
        "MC3-18": "mc3_18", 
        "R2Plus1D-18": "r2plus1d_18"
    }
    
    selected_model_name = st.sidebar.selectbox(
        "Select Model Architecture:",
        options=list(model_options.keys()),
        index=0
    )
    
    model_key = model_options[selected_model_name]
    
    # Load model
    with st.spinner(f"Loading {selected_model_name} model..."):
        model = load_model(model_key)
    
    if model is None:
        st.error("Failed to load model. Please check the error messages above.")
        return
    
    # Main content
    st.markdown("### 📹 Upload a Video")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to analyze for human actions"
    )
    
    # Create synthetic video option
    st.markdown("### 🎭 Or Create a Test Video")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        action_type = st.selectbox(
            "Action Type:",
            ["walking", "jumping", "static"],
            help="Type of action to simulate"
        )
    
    with col2:
        duration = st.slider(
            "Duration (seconds):",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5
        )
    
    with col3:
        if st.button("🎬 Create Test Video"):
            with st.spinner("Creating synthetic video..."):
                try:
                    test_video_path = "data/synthetic_test.mp4"
                    create_synthetic_video(test_video_path, action_type, duration)
                    
                    # Display the created video
                    st.video(test_video_path)
                    
                    # Make prediction on synthetic video
                    with st.spinner("Analyzing video..."):
                        predictions = model.predict_from_file(test_video_path, top_k=5)
                        display_prediction_results(predictions, model_key)
                        
                except Exception as e:
                    st.error(f"Failed to create or analyze test video: {e}")
    
    # Process uploaded file
    if uploaded_file is not None:
        # Display video info
        st.markdown("### 📊 Video Information")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Size", f"{uploaded_file.size / (1024*1024):.1f} MB")
        with col2:
            st.metric("File Type", uploaded_file.type)
        with col3:
            st.metric("File Name", uploaded_file.name)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Display video
            st.markdown("### 🎥 Your Video")
            st.video(tmp_file_path)
            
            # Analyze button
            if st.button("🔍 Analyze Video", type="primary"):
                with st.spinner("Analyzing video for actions..."):
                    try:
                        # Make prediction
                        predictions = model.predict_from_file(tmp_file_path, top_k=5)
                        
                        # Display results
                        display_prediction_results(predictions, model_key)
                        
                        # Additional analysis
                        st.markdown("### 📈 Additional Analysis")
                        
                        # Extract frames for visualization
                        frames = VideoAnalyzer.extract_frames(tmp_file_path, max_frames=8)
                        
                        if frames:
                            st.markdown("#### Key Frames:")
                            cols = st.columns(4)
                            for i, frame in enumerate(frames[:4]):
                                with cols[i % 4]:
                                    st.image(frame, caption=f"Frame {i+1}", use_column_width=True)
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        logger.error(f"Analysis error: {e}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎬 Action Recognition in Videos | Powered by PyTorch & Streamlit</p>
        <p>Built with modern 3D CNN architectures for accurate human action recognition</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
