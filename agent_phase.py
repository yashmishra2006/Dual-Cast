import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentPhaseClassifier:
    """
    Agent Selection Phase classifier for DualCast eSports commentary system.
    
    Detects agent selection status in Valorant agent selection phase:
    - Agent identity (Jett, Sage, Phoenix, etc.)
    - Status: 'hovered', 'selected', or 'detected'
    """
    
    def __init__(self, 
                 model_path: str = "resnet_agent.pt",
                 device: Optional[str] = None,
                 input_size: Tuple[int, int] = (224, 224),
                 confidence_threshold: float = 0.5,
                 output_dir: str = ".",
                 enable_json_output: bool = True):
        """
        Initialize AgentPhaseClassifier.
        
        Args:
            model_path: Path to the trained ResNet-18 model
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
        self.output_filename = "agent_phase_state.json"
        
        # Create output directory if it doesn't exist
        if self.enable_json_output:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Valorant agent names (adjust based on your training data)
        self.agent_names = [
            'Jett', 'Sage', 'Phoenix', 'Sova', 'Viper', 'Cypher', 'Reyna', 'Killjoy',
            'Breach', 'Omen', 'Brimstone', 'Raze', 'Skye', 'Yoru', 'Astra', 'KAY/O',
            'Chamber', 'Neon', 'Fade', 'Harbor', 'Gekko', 'Deadlock', 'Iso', 'Clove'
        ]
        
        # Status mapping
        self.status_mapping = {
            0: 'hovered',
            1: 'selected',
            2: 'detected'
        }
        
        # Load model
        self.model = self._load_model()
        
        # Setup preprocessing transforms
        self.transform = self._setup_transforms()
        
        logger.info("AgentPhaseClassifier initialized successfully")
        if self.enable_json_output:
            logger.info(f"JSON output enabled: {os.path.join(self.output_dir, self.output_filename)}")
    
    def _load_model(self) -> nn.Module:
        """Load the trained ResNet-18 model."""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load model
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different model save formats
            if isinstance(checkpoint, dict):
                # If model was saved as state_dict, reconstruct ResNet-18
                from torchvision.models import resnet18
                
                model = resnet18(pretrained=False)
                
                # Calculate output size: num_agents * num_statuses
                num_classes = len(self.agent_names) * len(self.status_mapping)
                
                # Modify final layer for agent + status classification
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                
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
            
            logger.info(f"ResNet-18 model loaded successfully from {self.model_path}")
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
            Preprocessed tensor ready for model input
        """
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            tensor = self.transform(frame_rgb)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            raise
    
    def _decode_prediction(self, prediction_idx: int) -> Tuple[str, str]:
        """
        Decode model prediction to agent name and status.
        
        Args:
            prediction_idx: Model prediction index
            
        Returns:
            Tuple of (agent_name, status)
        """
        try:
            # Assuming output is structured as: [agent0_status0, agent0_status1, agent0_status2, agent1_status0, ...]
            num_statuses = len(self.status_mapping)
            
            agent_idx = prediction_idx // num_statuses
            status_idx = prediction_idx % num_statuses
            
            agent_name = self.agent_names[agent_idx] if agent_idx < len(self.agent_names) else "Unknown"
            status = self.status_mapping[status_idx]
            
            return agent_name, status
            
        except Exception as e:
            logger.error(f"Error decoding prediction: {e}")
            return "Unknown", "detected"
    
    def _write_json_output(self, result: Dict[str, Any], frame_count: Optional[int] = None):
        """
        Write agent detection result to JSON file.
        
        Args:
            result: Detection result dictionary
            frame_count: Current frame number
        """
        if not self.enable_json_output:
            return
            
        try:
            # Create output data structure
            output_data = {
                **result,  # Include agent, status, confidence
                "timestamp": datetime.now().isoformat(),
                "model_info": {
                    "architecture": "ResNet-18",
                    "confidence_threshold": self.confidence_threshold,
                    "input_size": self.input_size,
                    "total_agents": len(self.agent_names),
                    "total_statuses": len(self.status_mapping)
                }
            }
            
            # Add optional data
            if frame_count is not None:
                output_data["frame_count"] = frame_count
            
            # Write to JSON file
            filepath = os.path.join(self.output_dir, self.output_filename)
            
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2)
                
            logger.debug(f"Agent phase JSON output written to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to write JSON output: {e}")
    
    def process_agent_frame(self, frame: np.ndarray, frame_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Process agent selection frame and detect agent + status.
        
        Args:
            frame: OpenCV frame (BGR format)
            frame_count: Optional frame number for JSON output
            
        Returns:
            Dictionary with agent, status, and confidence
        """
        try:
            # Preprocess frame
            input_tensor = self._preprocess_frame(frame)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                # Decode prediction
                agent_name, status = self._decode_prediction(predicted_class.item())
                
                # Create result
                result = {
                    'agent': agent_name,
                    'status': status,
                    'confidence': confidence.item()
                }
                
                # Check confidence threshold
                if confidence.item() < self.confidence_threshold:
                    logger.warning(f"Low confidence agent prediction: {confidence.item():.3f}")
                    result['low_confidence'] = True
                
                # Write JSON output
                self._write_json_output(result, frame_count)
                
                logger.debug(f"Agent prediction: {agent_name} - {status} (confidence: {confidence.item():.3f})")
                return result
                
        except Exception as e:
            logger.error(f"Error during agent frame processing: {e}")
            # Return default result on error
            default_result = {
                'agent': 'Unknown',
                'status': 'detected',
                'confidence': 0.0,
                'error': True
            }
            self._write_json_output(default_result, frame_count)
            return default_result
    
    def process_agent_frame_detailed(self, frame: np.ndarray, frame_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Process agent selection frame with detailed predictions for all agents.
        
        Args:
            frame: OpenCV frame (BGR format)
            frame_count: Optional frame number for JSON output
            
        Returns:
            Dictionary with top prediction and all agent probabilities
        """
        try:
            # Preprocess frame
            input_tensor = self._preprocess_frame(frame)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                # Decode top prediction
                agent_name, status = self._decode_prediction(predicted_class.item())
                
                # Get detailed probabilities for all agents
                detailed_probs = {}
                for i, prob in enumerate(probabilities[0]):
                    agent, stat = self._decode_prediction(i)
                    if agent not in detailed_probs:
                        detailed_probs[agent] = {}
                    detailed_probs[agent][stat] = prob.item()
                
                # Create detailed result
                result = {
                    'agent': agent_name,
                    'status': status,
                    'confidence': confidence.item(),
                    'all_agent_probabilities': detailed_probs
                }
                
                # Write JSON output
                self._write_json_output(result, frame_count)
                
                return result
                
        except Exception as e:
            logger.error(f"Error during detailed agent frame processing: {e}")
            # Return default result on error
            default_result = {
                'agent': 'Unknown',
                'status': 'detected',
                'confidence': 0.0,
                'error': True
            }
            self._write_json_output(default_result, frame_count)
            return default_result
    
    def should_run(self, frame_count: int) -> bool:
        """
        Determine if agent classification should run for this frame.
        
        Args:
            frame_count: Current frame number
            
        Returns:
            True if classification should run (every 10 frames)
        """
        return frame_count % 10 == 0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "device": str(self.device),
            "input_size": self.input_size,
            "confidence_threshold": self.confidence_threshold,
            "output_dir": self.output_dir,
            "output_filename": self.output_filename,
            "json_output_enabled": self.enable_json_output,
            "agent_names": self.agent_names,
            "status_mapping": self.status_mapping,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "model_architecture": "ResNet-18"
        }
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold."""
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            logger.info(f"Confidence threshold updated to {threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
    
    def toggle_json_output(self, enable: bool):
        """Enable or disable JSON output."""
        self.enable_json_output = enable
        logger.info(f"JSON output {'enabled' if enable else 'disabled'}")
    
    def set_output_dir(self, output_dir: str):
        """Change output directory for JSON files."""
        self.output_dir = output_dir
        if self.enable_json_output:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Output directory changed to: {self.output_dir}")
    
    def get_supported_agents(self) -> list:
        """Get list of supported agent names."""
        return self.agent_names.copy()
    
    def get_supported_statuses(self) -> list:
        """Get list of supported status types."""
        return list(self.status_mapping.values())


# Example usage and testing
if __name__ == "__main__":
    # Initialize agent classifier
    try:
        agent_classifier = AgentPhaseClassifier(
            model_path="resnet_agent.pt",
            confidence_threshold=0.6,
            output_dir="./agent_outputs",
            enable_json_output=True
        )
        
        print("=== Agent Phase Classifier Information ===")
        info = agent_classifier.get_model_info()
        for key, value in info.items():
            if key == "agent_names":
                print(f"{key}: {len(value)} agents supported")
            else:
                print(f"{key}: {value}")
        
        print(f"\nSupported agents: {', '.join(agent_classifier.get_supported_agents()[:5])}...")
        print(f"Supported statuses: {', '.join(agent_classifier.get_supported_statuses())}")
        
        # Test with dummy frame
        print("\n=== Testing with dummy frame ===")
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Basic agent detection
        result = agent_classifier.process_agent_frame(dummy_frame, frame_count=50)
        print(f"Agent detection result: {result}")
        
        # Detailed agent detection
        detailed_result = agent_classifier.process_agent_frame_detailed(dummy_frame, frame_count=51)
        print(f"Detailed result - Agent: {detailed_result['agent']}, Status: {detailed_result['status']}")
        
        # Test frame scheduling
        print("\n=== Testing frame scheduling ===")
        for i in range(15):
            should_run = agent_classifier.should_run(i)
            print(f"Frame {i}: {'RUN' if should_run else 'SKIP'}")
        
        print(f"\nJSON output file: {agent_classifier.output_filename}")
        
    except Exception as e:
        print(f"Failed to initialize agent classifier: {e}")
        print("Make sure 'resnet_agent.pt' exists in the current directory")