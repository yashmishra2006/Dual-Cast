"""
DualCast - Buy Phase Detection Module
====================================

This module handles weapon detection and classification during Valorant's buy phase.
Uses YOLO for weapon bounding box detection and ResNet-18 for weapon classification.

Dependencies:
- ultralytics (YOLO)
- torch, torchvision (ResNet)
- opencv-python (cv2)
- PIL (Pillow)

Models required:
- buy_1.pt (YOLO model for weapon detection)
- buy_2.pt (ResNet-18 model for weapon classification)
"""

import json
import os
import tempfile
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeaponClassifier(nn.Module):
    """ResNet-18 based weapon classifier"""
    
    def __init__(self, num_classes=10):  # Adjust based on your weapon classes
        super(WeaponClassifier, self).__init__()
        # Load pretrained ResNet-18
        from torchvision.models import resnet18
        self.backbone = resnet18(pretrained=False)
        # Replace final layer for weapon classification
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)


class BuyPhaseDetector:
    """Main class for buy phase weapon detection and classification"""
    
    def __init__(self, yolo_model_path="models/buy_1.pt", resnet_model_path="models/buy_2.pt"):
        """
        Initialize the buy phase detector
        
        Args:
            yolo_model_path (str): Path to YOLO model file
            resnet_model_path (str): Path to ResNet model file
        """
        self.yolo_model_path = yolo_model_path
        self.resnet_model_path = resnet_model_path
        self.output_file = "buy_phase_state.json"
        
        # Model placeholders
        self.yolo_model = None
        self.resnet_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Weapon class mapping (adjust based on your model's classes)
        self.weapon_classes = {
            0: 'ares',
            1: 'bucky',
            2: 'bulldog',
            3: 'classic',
            4: 'frenzy',
            5: 'guardian',
            6: 'ghost',
            7: 'heavy shield',
            8: 'judge',
            9: 'light shield',
            10: 'marshal',
            11: 'odin',
            12: 'operator',
            13: 'outlaw',
            14: 'phantom',
            15: 'regen shield',
            16: 'sheriff',
            17: 'shorty',
            18: 'spectre',
            19: 'stinger',
            20: 'vandal'
        }
        
        # Status mapping based on detection confidence and context
        self.status_thresholds = {
            "owned": 0.8,
            "hovered": 0.6,
            "available": 0.4
        }
        
        # Image preprocessing for ResNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize models
        self._load_models()
    
    def _load_models(self):
        """Load YOLO and ResNet models"""
        try:
            # Load YOLO model
            if os.path.exists(self.yolo_model_path):
                self.yolo_model = YOLO(self.yolo_model_path)
                logger.info(f"YOLO model loaded from {self.yolo_model_path}")
            else:
                logger.warning(f"YOLO model not found: {self.yolo_model_path}")
                self.yolo_model = None
            
            # Load ResNet model
            if os.path.exists(self.resnet_model_path):
                self.resnet_model = WeaponClassifier(num_classes=len(self.weapon_classes))
                self.resnet_model.load_state_dict(torch.load(self.resnet_model_path, map_location=self.device))
                self.resnet_model.to(self.device)
                self.resnet_model.eval()
                logger.info(f"ResNet model loaded from {self.resnet_model_path}")
            else:
                logger.warning(f"ResNet model not found: {self.resnet_model_path}")
                self.resnet_model = None
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.yolo_model = None
            self.resnet_model = None
    
    def _detect_weapons(self, frame):
        """
        Detect weapon bounding boxes using YOLO
        
        Args:
            frame (numpy.ndarray): Input frame from OpenCV
            
        Returns:
            list: List of detected bounding boxes
        """
        if self.yolo_model is None:
            logger.warning("YOLO model not loaded")
            return []
        
        try:
            # Run YOLO detection
            results = self.yolo_model(frame)
            
            # Extract bounding boxes
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence)
                        })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in YOLO detection: {e}")
            return []
    
    def _classify_weapon(self, weapon_crop):
        """
        Classify weapon using ResNet
        
        Args:
            weapon_crop (numpy.ndarray): Cropped weapon image
            
        Returns:
            tuple: (weapon_name, confidence)
        """
        if self.resnet_model is None:
            logger.warning("ResNet model not loaded")
            return "Unknown", 0.0
        
        try:
            # Convert BGR to RGB
            weapon_crop_rgb = cv2.cvtColor(weapon_crop, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(weapon_crop_rgb)
            
            # Apply transforms
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.resnet_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # Get predicted class and confidence
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item()
                
                weapon_name = self.weapon_classes.get(predicted_class, "Unknown")
                
                return weapon_name, confidence
                
        except Exception as e:
            logger.error(f"Error in weapon classification: {e}")
            return "Unknown", 0.0
    
    def _determine_status(self, detection_confidence, classification_confidence):
        """
        Determine weapon status based on confidences
        
        Args:
            detection_confidence (float): YOLO detection confidence
            classification_confidence (float): ResNet classification confidence
            
        Returns:
            str: Status ("owned", "hovered", "available")
        """
        # Combine confidences (you may want to adjust this logic)
        combined_confidence = (detection_confidence + classification_confidence) / 2
        
        if combined_confidence >= self.status_thresholds["owned"]:
            return "owned"
        elif combined_confidence >= self.status_thresholds["hovered"]:
            return "hovered"
        else:
            return "available"
    
    def _write_output(self, results):
        """
        Write detection results to JSON file
        
        Args:
            results (list): List of detection results
        """
        try:
            with open(self.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.debug(f"Results written to {self.output_file}")
        except Exception as e:
            logger.error(f"Error writing output file: {e}")
    
    def process_buy_frame(self, frame):
        """
        Process a single frame for buy phase detection
        
        Args:
            frame (numpy.ndarray): Input frame from OpenCV
            
        Returns:
            list: List of weapon detection results
        """
        results = []
        
        try:
            # Step 1: Detect weapons using YOLO
            weapon_detections = self._detect_weapons(frame)
            
            # Step 2: Classify each detected weapon
            for detection in weapon_detections:
                bbox = detection['bbox']
                detection_conf = detection['confidence']
                
                # Crop weapon from frame
                x1, y1, x2, y2 = bbox
                weapon_crop = frame[y1:y2, x1:x2]
                
                # Skip if crop is too small
                if weapon_crop.shape[0] < 10 or weapon_crop.shape[1] < 10:
                    continue
                
                # Classify weapon
                weapon_name, classification_conf = self._classify_weapon(weapon_crop)
                
                # Determine status
                status = self._determine_status(detection_conf, classification_conf)
                
                # Add to results
                results.append({
                    "weapon": weapon_name,
                    "status": status,
                    "confidence": round(classification_conf, 2)
                })
            
            # Step 3: Write results to JSON file
            self._write_output(results)
            
            logger.info(f"Processed frame: {len(results)} weapons detected")
            
        except Exception as e:
            logger.error(f"Error processing buy frame: {e}")
            # Write empty results on error
            self._write_output([])
        
        return results


# Global detector instance
_detector = None


def _get_detector():
    """Get or create detector instance"""
    global _detector
    if _detector is None:
        _detector = BuyPhaseDetector()
    return _detector


def should_run(frame_count):
    """
    Determine if processing should run for this frame
    
    Args:
        frame_count (int): Current frame number
        
    Returns:
        bool: True if processing should run
    """
    # Run every 2 frames to balance performance and responsiveness
    return frame_count % 2 == 0


def process_buy_frame(frame):
    """
    Main entry point for buy phase processing
    
    Args:
        frame (numpy.ndarray): Input frame from OpenCV
        
    Returns:
        list: List of weapon detection results
    """
    detector = _get_detector()
    return detector.process_buy_frame(frame)


# Example usage and testing
if __name__ == "__main__":
    # Test the module
    import numpy as np
    
    # Create a dummy frame for testing
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Test should_run function
    print("Testing should_run function:")
    for i in range(5):
        print(f"Frame {i}: {should_run(i)}")
    
    # Test process_buy_frame (will handle missing models gracefully)
    print("\nTesting process_buy_frame:")
    results = process_buy_frame(test_frame)
    print(f"Results: {results}")
    
    # Check if output file was created
    if os.path.exists("buy_phase_state.json"):
        with open("buy_phase_state.json", 'r') as f:
            file_contents = json.load(f)
        print(f"Output file contents: {file_contents}")