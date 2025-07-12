import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
import os
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhaseClassifier:
    """
    Phase classifier for DualCast eSports commentary system.
    
    Detects game phases in Valorant gameplay:
    - 'agent': Agent selection phase
    - 'buy': Buy phase
    - 'game': Active gameplay phase
    """
    
    def __init__(self, 
                 model_path: str = "phase_classifier.pt",
                 device: Optional[str] = None,
                 input_size: Tuple[int, int] = (224, 224),
                 confidence_threshold: float = 0.5,
                 output_dir: str = ".",
                 enable_json_output: bool = True):
        """
        Initialize PhaseClassifier.
        
        Args:
            model_path: Path to the trained PyTorch model
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            input_size: Input image size expected by model (width, height)
            confidence_threshold: Minimum confidence for predictions
            output_dir: Directory to save JSON output files
            enable_json_output: Whether to write JSON files after each prediction
        """
        self.model_path = model_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.output_dir = output_dir
        self.enable_json_output = enable_json_output
        
        # Create output directory if it doesn't exist
        if self.enable_json_output:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Phase mapping and corresponding file names
        self.phase_mapping = {
            0: 'agent',
            1: 'buy', 
            2: 'game'
        }
        
        self.phase_files = {
            'agent': 'agent_phase_state.json',
            'buy': 'buy_phase_state.json',
            'game': 'game_phase_state.json'
        }
        
        # Load model
        self.model = self._load_model()
        
        # Setup preprocessing transforms
        self.transform = self._setup_transforms()
        
        logger.info("PhaseClassifier initialized successfully")
        if self.enable_json_output:
            logger.info(f"JSON output enabled in directory: {self.output_dir}")
    
    def _load_model(self) -> nn.Module:
        """Load the trained PyTorch model."""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load model
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different model save formats
            if isinstance(checkpoint, dict):
                # If model was saved as state_dict, reconstruct ResNet50
                from torchvision.models import resnet50
                
                model = resnet50(pretrained=False)
                # Modify final layer for 3 classes (agent, buy, game)
                model.fc = nn.Linear(model.fc.in_features, 3)
                
                # Load state dict
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                # Model was saved as complete model
                model = checkpoint
            
            model.eval()
            model.to(self.device)
            
            logger.info(f"ResNet50 model loaded successfully from {self.model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_transforms(self) -> transforms.Compose:
        """Setup preprocessing transforms matching training pipeline."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet standard
                std=[0.229, 0.224, 0.225]   # ImageNet standard
            )
        ])
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for model input.
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns: