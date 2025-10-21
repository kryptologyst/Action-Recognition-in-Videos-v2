"""
Configuration management for the action recognition project.

This module handles loading and managing configuration settings from YAML files
and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Configuration for the action recognition model."""
    name: str = "r3d_18"
    device: Optional[str] = None
    num_classes: int = 400
    max_frames: int = 32
    input_size: tuple = (112, 112)
    batch_size: int = 1


@dataclass
class DataConfig:
    """Configuration for data processing."""
    data_dir: str = "data"
    labels_file: str = "kinetics_labels.txt"
    test_video: str = "test_video.mp4"
    synthetic_video_duration: float = 3.0
    synthetic_video_fps: int = 30


@dataclass
class UIConfig:
    """Configuration for the user interface."""
    title: str = "Action Recognition in Videos"
    description: str = "Identify human actions in video clips using 3D CNNs"
    max_file_size_mb: int = 100
    supported_formats: list = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [".mp4", ".avi", ".mov", ".mkv"]


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = "logs/action_recognition.log"


@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig
    data: DataConfig
    ui: UIConfig
    logging: LoggingConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AppConfig':
        """Create AppConfig from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            ui=UIConfig(**config_dict.get('ui', {})),
            logging=LoggingConfig(**config_dict.get('logging', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert AppConfig to dictionary."""
        return {
            'model': asdict(self.model),
            'data': asdict(self.data),
            'ui': asdict(self.ui),
            'logging': asdict(self.logging)
        }


class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (default: config/config.yaml)
        """
        self.config_path = Path(config_path or "config/config.yaml")
        self.config: Optional[AppConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file or create default."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                self.config = AppConfig.from_dict(config_dict)
            except Exception as e:
                print(f"Failed to load config from {self.config_path}: {e}")
                self._create_default_config()
        else:
            self._create_default_config()
            self._save_config()
    
    def _create_default_config(self) -> None:
        """Create default configuration."""
        self.config = AppConfig(
            model=ModelConfig(),
            data=DataConfig(),
            ui=UIConfig(),
            logging=LoggingConfig()
        )
    
    def _save_config(self) -> None:
        """Save current configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)
    
    def get_config(self) -> AppConfig:
        """Get current configuration."""
        if self.config is None:
            self._create_default_config()
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values."""
        if self.config is None:
            self._create_default_config()
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self._save_config()
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._load_config()


# Global configuration instance
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get the global application configuration."""
    return config_manager.get_config()


def update_config(**kwargs) -> None:
    """Update the global application configuration."""
    config_manager.update_config(**kwargs)
