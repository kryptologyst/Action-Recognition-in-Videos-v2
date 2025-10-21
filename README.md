# Action Recognition in Videos

A comprehensive implementation of action recognition in videos using 3D Convolutional Neural Networks (CNNs). This project provides both web and command-line interfaces for analyzing human actions in video clips.

## Features

- **Multiple Model Architectures**: Support for R3D-18, MC3-18, and R2Plus1D-18 models
- **Modern Implementation**: Updated to use current PyTorch best practices
- **Web Interface**: User-friendly Streamlit app for easy interaction
- **Command Line Tool**: CLI for batch processing and automation
- **Synthetic Data Generation**: Create test videos for development and testing
- **Comprehensive Testing**: Full test suite with unit and integration tests
- **Configuration Management**: YAML-based configuration system
- **Type Safety**: Full type hints and modern Python practices

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Action-Recognition-in-Videos-v2.git
   cd Action-Recognition-in-Videos-v2
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web interface**:
   ```bash
   streamlit run web_app/app.py
   ```

4. **Or use the command line**:
   ```bash
   python src/cli.py predict your_video.mp4
   ```

## 📁 Project Structure

```
0222_Action_recognition_in_videos/
├── src/                          # Source code
│   ├── action_recognition.py     # Core action recognition module
│   ├── config.py                 # Configuration management
│   └── cli.py                    # Command line interface
├── web_app/                      # Web interface
│   └── app.py                    # Streamlit application
├── tests/                        # Test suite
│   └── test_action_recognition.py
├── config/                       # Configuration files
│   └── config.yaml
├── data/                         # Data directory (created automatically)
├── models/                       # Model storage (created automatically)
├── logs/                         # Log files (created automatically)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Usage Examples

### Web Interface

Launch the Streamlit app for an interactive experience:

```bash
streamlit run web_app/app.py
```

Features:
- Upload video files for analysis
- Create synthetic test videos
- View predictions with confidence scores
- Visualize key frames
- Switch between different model architectures

### Command Line Interface

#### Analyze a Single Video

```bash
python src/cli.py predict video.mp4
```

#### Use Different Model Architecture

```bash
python src/cli.py predict video.mp4 --model mc3_18
```

#### Create Synthetic Test Video

```bash
python src/cli.py create-test --output test.mp4 --action jumping --duration 5.0
```

#### Batch Process Multiple Videos

```bash
python src/cli.py batch videos/ --output-dir results/ --model r2plus1d_18
```

#### Save Results to File

```bash
python src/cli.py predict video.mp4 --output results.txt
```

### Python API

```python
from src.action_recognition import ActionRecognitionModel

# Initialize model
model = ActionRecognitionModel(model_name="r3d_18")

# Predict actions
predictions = model.predict_from_file("video.mp4", top_k=5)

# Display results
for action, confidence in predictions:
    print(f"{action}: {confidence:.3f}")
```

## Model Architectures

### R3D-18
- **Architecture**: 3D ResNet with 18 layers
- **Best for**: General action recognition tasks
- **Input**: 32 frames × 112×112 pixels

### MC3-18
- **Architecture**: Mixed Convolution with 18 layers
- **Best for**: Efficient processing with good accuracy
- **Input**: 32 frames × 112×112 pixels

### R2Plus1D-18
- **Architecture**: R(2+1)D with 18 layers
- **Best for**: Better temporal modeling
- **Input**: 32 frames × 112×112 pixels

## Configuration

The project uses YAML configuration files for easy customization:

```yaml
# config/config.yaml
model:
  name: "r3d_18"
  device: null  # Auto-detect
  num_classes: 400
  max_frames: 32

data:
  data_dir: "data"
  synthetic_video_duration: 3.0

ui:
  title: "Action Recognition in Videos"
  max_file_size_mb: 100
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src/

# Run specific test file
python tests/test_action_recognition.py
```

## Supported Video Formats

- **Input**: MP4, AVI, MOV, MKV, WMV, FLV
- **Output**: MP4 (for synthetic videos)
- **Recommended**: MP4 with H.264 encoding

## 🔧 Development

### Code Quality

The project follows modern Python best practices:

- **Type Hints**: Full type annotations
- **Documentation**: Comprehensive docstrings
- **Style**: PEP 8 compliance
- **Testing**: Unit and integration tests
- **Logging**: Structured logging throughout

### Running Tests

```bash
# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Run tests
pytest tests/ -v
```

## Performance

### Hardware Requirements

- **Minimum**: CPU-only (slower inference)
- **Recommended**: GPU with CUDA support
- **Memory**: 4GB+ RAM, 2GB+ VRAM (if using GPU)

### Optimization Tips

1. **Use GPU**: Automatically detected if available
2. **Batch Processing**: Use CLI for multiple videos
3. **Video Preprocessing**: Shorter videos process faster
4. **Model Selection**: MC3-18 is faster than R3D-18

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in config
   - Use CPU instead of GPU
   - Process shorter video clips

2. **Video Format Issues**:
   - Convert to MP4 with H.264
   - Check file permissions
   - Verify video file integrity

3. **Model Loading Errors**:
   - Check internet connection (for downloading weights)
   - Verify PyTorch installation
   - Clear model cache and retry

### Getting Help

- Check the logs in `logs/action_recognition.log`
- Run with `--verbose` flag for detailed output
- Ensure all dependencies are installed correctly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- **PyTorch Team**: For the excellent deep learning framework
- **Torchvision**: For pre-trained video models
- **Kinetics Dataset**: For the training data
- **Streamlit**: For the web interface framework

## Future Enhancements

- [ ] Real-time video stream processing
- [ ] Multi-person action detection
- [ ] Custom model fine-tuning
- [ ] Advanced visualization features
- [ ] REST API endpoint
- [ ] Docker containerization
- [ ] Model quantization for mobile deployment


# Action-Recognition-in-Videos-v2
