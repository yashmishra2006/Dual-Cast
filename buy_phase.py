"""
DualCast - Enhanced Buy Phase Detection Module
=============================================

This module handles weapon detection and classification during Valorant's buy phase.
Uses YOLO for weapon detection and ResNet-18 for weapon classification with advanced
tracking, smoothing, and annotation capabilities.

Dependencies:
- ultralytics (YOLO)
- torch, torchvision (ResNet)
- opencv-python (cv2)
- PIL (Pillow)
- numpy

Models required:
- buy_1.pt (YOLO model for weapon detection)
- buy_2.pt (ResNet-18 model for weapon classification)
"""

import json
import os
import tempfile
import shutil
import uuid
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionTracker:
    """Custom tracker for managing object IDs and smoothing detections"""
    
    def __init__(self, max_disappeared=5, max_distance=50):
        self.next_id = 0
        self.objects = {}  # id -> {bbox, confidence, class, last_seen}
        self.disappeared = {}  # id -> frames_disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Smoothing parameters
        self.bbox_history = defaultdict(lambda: deque(maxlen=5))
        self.confidence_history = defaultdict(lambda: deque(maxlen=5))
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1, y1, x2, y2 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x1_2)
        yi1 = max(y1, y1_2)
        xi2 = min(x2, x2_2)
        yi2 = min(y2, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _calculate_distance(self, box1, box2):
        """Calculate center distance between two bounding boxes"""
        c1_x = (box1[0] + box1[2]) / 2
        c1_y = (box1[1] + box1[3]) / 2
        c2_x = (box2[0] + box2[2]) / 2
        c2_y = (box2[1] + box2[3]) / 2
        
        return np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
    
    def _smooth_bbox(self, object_id, bbox):
        """Apply moving average to bounding box coordinates"""
        self.bbox_history[object_id].append(bbox)
        history = list(self.bbox_history[object_id])
        
        if len(history) == 1:
            return bbox
        
        # Calculate weighted average (recent frames have higher weight)
        weights = np.linspace(0.5, 1.0, len(history))
        weights = weights / weights.sum()
        
        smoothed = np.zeros(4)
        for i, (weight, box) in enumerate(zip(weights, history)):
            smoothed += weight * np.array(box)
        
        return smoothed.astype(int).tolist()
    
    def _smooth_confidence(self, object_id, confidence):
        """Apply moving average to confidence scores"""
        self.confidence_history[object_id].append(confidence)
        history = list(self.confidence_history[object_id])
        
        if len(history) == 1:
            return confidence
        
        # Simple moving average for confidence
        return sum(history) / len(history)
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections (list): List of detections with bbox, confidence, class
            
        Returns:
            dict: Updated objects with IDs
        """
        if not detections:
            # Mark all objects as disappeared
            for obj_id in list(self.objects.keys()):
                self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister_object(obj_id)
            return {}
        
        # If no existing objects, register all detections
        if not self.objects:
            for detection in detections:
                self._register_object(detection)
        else:
            # Match detections to existing objects
            self._match_detections(detections)
        
        # Clean up disappeared objects
        self._cleanup_disappeared()
        
        return self.objects.copy()
    
    def _register_object(self, detection):
        """Register a new object"""
        obj_id = self.next_id
        self.objects[obj_id] = {
            'bbox': detection['bbox'],
            'confidence': detection['confidence'],
            'class': detection.get('class', 'unknown'),
            'last_seen': 0
        }
        self.next_id += 1
        logger.debug(f"Registered new object ID: {obj_id}")
    
    def _deregister_object(self, obj_id):
        """Remove an object from tracking"""
        if obj_id in self.objects:
            del self.objects[obj_id]
        if obj_id in self.disappeared:
            del self.disappeared[obj_id]
        if obj_id in self.bbox_history:
            del self.bbox_history[obj_id]
        if obj_id in self.confidence_history:
            del self.confidence_history[obj_id]
        logger.debug(f"Deregistered object ID: {obj_id}")
    
    def _match_detections(self, detections):
        """Match new detections to existing objects"""
        # Calculate IoU matrix
        object_ids = list(self.objects.keys())
        iou_matrix = np.zeros((len(object_ids), len(detections)))
        
        for i, obj_id in enumerate(object_ids):
            for j, detection in enumerate(detections):
                iou = self._calculate_iou(self.objects[obj_id]['bbox'], detection['bbox'])
                distance = self._calculate_distance(self.objects[obj_id]['bbox'], detection['bbox'])
                
                # Use IoU if good overlap, otherwise use distance
                if iou > 0.3:
                    iou_matrix[i, j] = iou
                elif distance < self.max_distance:
                    iou_matrix[i, j] = 0.1  # Low but non-zero score for distance match
        
        # Find best matches
        used_detection_indices = set()
        used_object_indices = set()
        
        # Sort by IoU score and assign matches
        matches = []
        for i in range(len(object_ids)):
            for j in range(len(detections)):
                if i not in used_object_indices and j not in used_detection_indices:
                    if iou_matrix[i, j] > 0.1:
                        matches.append((i, j, iou_matrix[i, j]))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # Update matched objects
        for obj_idx, det_idx, score in matches:
            if obj_idx not in used_object_indices and det_idx not in used_detection_indices:
                obj_id = object_ids[obj_idx]
                detection = detections[det_idx]
                
                # Update object with smoothed values
                smoothed_bbox = self._smooth_bbox(obj_id, detection['bbox'])
                smoothed_conf = self._smooth_confidence(obj_id, detection['confidence'])
                
                self.objects[obj_id].update({
                    'bbox': smoothed_bbox,
                    'confidence': smoothed_conf,
                    'class': detection.get('class', 'unknown'),
                    'last_seen': 0
                })
                
                # Reset disappeared counter
                if obj_id in self.disappeared:
                    del self.disappeared[obj_id]
                
                used_object_indices.add(obj_idx)
                used_detection_indices.add(det_idx)
        
        # Mark unmatched objects as disappeared
        for i, obj_id in enumerate(object_ids):
            if i not in used_object_indices:
                self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1
                self.objects[obj_id]['last_seen'] += 1
        
        # Register new detections
        for j, detection in enumerate(detections):
            if j not in used_detection_indices:
                self._register_object(detection)
    
    def _cleanup_disappeared(self):
        """Remove objects that have been disappeared for too long"""
        to_remove = []
        for obj_id, frames_disappeared in self.disappeared.items():
            if frames_disappeared > self.max_disappeared:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            self._deregister_object(obj_id)


class TemporalStabilizer:
    """Manages temporal smoothing across frames"""
    
    def __init__(self, buffer_size=3, stability_threshold=0.5):
        self.buffer_size = buffer_size
        self.stability_threshold = stability_threshold
        self.frame_buffer = deque(maxlen=buffer_size)
    
    def add_frame(self, detections):
        """Add a frame's detections to the buffer"""
        self.frame_buffer.append(detections)
    
    def get_stable_detections(self):
        """Get detections that are stable across frames"""
        if len(self.frame_buffer) < 2:
            return self.frame_buffer[-1] if self.frame_buffer else {}
        
        # Find consistent detections across frames
        stable_detections = {}
        current_frame = self.frame_buffer[-1]
        
        for obj_id, detection in current_frame.items():
            stability_score = self._calculate_stability(obj_id, detection)
            if stability_score >= self.stability_threshold:
                stable_detections[obj_id] = detection
        
        return stable_detections
    
    def _calculate_stability(self, obj_id, detection):
        """Calculate stability score for a detection"""
        if len(self.frame_buffer) < 2:
            return 1.0
        
        stability_scores = []
        current_bbox = detection['bbox']
        
        # Check against previous frames
        for i in range(len(self.frame_buffer) - 1):
            frame_detections = self.frame_buffer[i]
            if obj_id in frame_detections:
                prev_bbox = frame_detections[obj_id]['bbox']
                iou = self._calculate_iou(current_bbox, prev_bbox)
                stability_scores.append(iou)
        
        return np.mean(stability_scores) if stability_scores else 0.0
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1, y1, x2, y2 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1, x1_2)
        yi1 = max(y1, y1_2)
        xi2 = min(x2, x2_2)
        yi2 = min(y2, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


class WeaponClassifier(nn.Module):
    """ResNet-18 based weapon classifier"""
    
    def __init__(self, num_classes=21):
        super(WeaponClassifier, self).__init__()
        from torchvision.models import resnet18
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)


class BuyPhaseDetector:
    """Enhanced buy phase weapon detection and classification module"""
    
    def __init__(self, yolo_model_path="models/buy_1.pt", resnet_model_path="models/buy_2.pt"):
        """
        Initialize the buy phase detector
        
        Args:
            yolo_model_path (str): Path to YOLO model file
            resnet_model_path (str): Path to ResNet model file
        """
        self.yolo_model_path = yolo_model_path
        self.resnet_model_path = resnet_model_path
        
        # Model placeholders
        self.yolo_model = None
        self.resnet_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Tracking components
        self.tracker = DetectionTracker()
        self.stabilizer = TemporalStabilizer()
        
        # Temporary directory for crops
        self.temp_dir = tempfile.mkdtemp(prefix="buy_phase_crops_")
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.min_box_size = 20
        self.max_box_size = 500
        self.aspect_ratio_range = (0.5, 2.0)
        self.classification_threshold = 0.6
        
        # Weapon class mapping
        self.weapon_classes = {
            0: 'ares', 1: 'bucky', 2: 'bulldog', 3: 'classic', 4: 'frenzy',
            5: 'guardian', 6: 'ghost', 7: 'heavy shield', 8: 'judge',
            9: 'light shield', 10: 'marshal', 11: 'odin', 12: 'operator',
            13: 'outlaw', 14: 'phantom', 15: 'regen shield', 16: 'sheriff',
            17: 'shorty', 18: 'spectre', 19: 'stinger', 20: 'vandal'
        }
        
        # Status mapping
        self.status_mapping = {
            'hover-box': 'hovered',
            'owned-box': 'owned',
            'default': 'available'
        }
        
        # Image preprocessing
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
            
            # Load ResNet model
            if os.path.exists(self.resnet_model_path):
                self.resnet_model = WeaponClassifier(num_classes=len(self.weapon_classes))
                self.resnet_model.load_state_dict(torch.load(self.resnet_model_path, map_location=self.device))
                self.resnet_model.to(self.device)
                self.resnet_model.eval()
                logger.info(f"ResNet model loaded from {self.resnet_model_path}")
            else:
                logger.warning(f"ResNet model not found: {self.resnet_model_path}")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.yolo_model = None
            self.resnet_model = None
    
    def _validate_bbox(self, bbox, frame_shape):
        """Validate bounding box based on size and aspect ratio"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Check size constraints
        if width < self.min_box_size or height < self.min_box_size:
            return False
        if width > self.max_box_size or height > self.max_box_size:
            return False
        
        # Check aspect ratio
        aspect_ratio = width / height
        if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
            return False
        
        # Check if bbox is within frame bounds
        h, w = frame_shape[:2]
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return False
        
        return True
    
    def _detect_weapons(self, frame):
        """Detect weapon bounding boxes using YOLO with advanced filtering"""
        if self.yolo_model is None:
            logger.warning("YOLO model not loaded")
            return []
        
        try:
            results = self.yolo_model(frame)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
                        
                        bbox = [int(x1), int(y1), int(x2), int(y2)]
                        
                        # Apply filters
                        if (confidence >= self.confidence_threshold and
                            self._validate_bbox(bbox, frame.shape)):
                            
                            # Map YOLO class to status
                            yolo_class_names = self.yolo_model.names
                            yolo_class_name = yolo_class_names.get(class_id, 'default')
                            
                            detections.append({
                                'bbox': bbox,
                                'confidence': float(confidence),
                                'yolo_class': yolo_class_name
                            })
            
            logger.debug(f"YOLO detected {len(detections)} valid objects")
            return detections
            
        except Exception as e:
            logger.error(f"Error in YOLO detection: {e}")
            return []
    
    def _save_crop(self, crop, weapon_name, confidence):
        """Save weapon crop to temporary directory"""
        try:
            filename = f"{weapon_name}_{confidence:.2f}_{uuid.uuid4().hex[:8]}.jpg"
            filepath = os.path.join(self.temp_dir, filename)
            cv2.imwrite(filepath, crop)
            logger.debug(f"Saved crop: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving crop: {e}")
            return None
    
    def _classify_weapon(self, weapon_crop, save_crop=False):
        """Classify weapon using ResNet with optional crop saving"""
        if self.resnet_model is None:
            logger.warning("ResNet model not loaded")
            return "Unknown", 0.0
        
        try:
            # Convert BGR to RGB
            weapon_crop_rgb = cv2.cvtColor(weapon_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(weapon_crop_rgb)
            
            # Apply transforms
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.resnet_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item()
                
                weapon_name = self.weapon_classes.get(predicted_class, "Unknown")
                
                # Save crop if requested and confidence is high enough
                if save_crop and confidence >= self.classification_threshold:
                    self._save_crop(weapon_crop, weapon_name, confidence)
                
                # Clear GPU memory
                del input_tensor, outputs, probabilities
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return weapon_name, confidence
                
        except Exception as e:
            logger.error(f"Error in weapon classification: {e}")
            return "Unknown", 0.0
    
    def _determine_status(self, yolo_class, classification_confidence):
        """Determine weapon status based on YOLO class and confidence"""
        if classification_confidence < self.classification_threshold:
            return "available"
        
        return self.status_mapping.get(yolo_class, "available")
    
    def _annotate_frame(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        annotated_frame = frame.copy()
        
        for obj_id, detection in detections.items():
            bbox = detection['bbox']
            confidence = detection['confidence']
            weapon_class = detection.get('class', 'Unknown')
            status = detection.get('status', 'available')
            
            x1, y1, x2, y2 = bbox
            
            # Choose color based on status
            colors = {
                'owned': (0, 255, 0),      # Green
                'hovered': (0, 255, 255),  # Yellow
                'available': (255, 0, 0)   # Blue
            }
            color = colors.get(status, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"ID:{obj_id} {weapon_class} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame
    
    def process_frame(self, frame, annotate=False, save_crops=False):
        """
        Process a single frame for buy phase detection
        
        Args:
            frame (numpy.ndarray): Input frame from OpenCV
            annotate (bool): Whether to return annotated frame
            save_crops (bool): Whether to save weapon crops
            
        Returns:
            dict: Detection results with optional annotated frame
        """
        try:
            # Step 1: Detect weapons using YOLO
            raw_detections = self._detect_weapons(frame)
            
            # Step 2: Classify each detected weapon
            classified_detections = []
            for detection in raw_detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                
                # Crop weapon from frame
                weapon_crop = frame[y1:y2, x1:x2]
                
                # Skip if crop is too small
                if weapon_crop.shape[0] < 10 or weapon_crop.shape[1] < 10:
                    continue
                
                # Classify weapon
                weapon_name, classification_conf = self._classify_weapon(
                    weapon_crop, save_crop=save_crops
                )
                
                # Determine status
                status = self._determine_status(
                    detection['yolo_class'], classification_conf
                )
                
                classified_detections.append({
                    'bbox': bbox,
                    'confidence': classification_conf,
                    'class': weapon_name,
                    'status': status,
                    'yolo_class': detection['yolo_class']
                })
            
            # Step 3: Update tracker
            tracked_objects = self.tracker.update(classified_detections)
            
            # Step 4: Apply temporal stabilization
            self.stabilizer.add_frame(tracked_objects)
            stable_detections = self.stabilizer.get_stable_detections()
            
            # Step 5: Prepare results
            results = {
                'detections': [],
                'tracked_objects': stable_detections,
                'frame_info': {
                    'total_detections': len(classified_detections),
                    'tracked_objects': len(tracked_objects),
                    'stable_objects': len(stable_detections)
                }
            }
            
            # Convert to output format
            for obj_id, detection in stable_detections.items():
                results['detections'].append({
                    'id': obj_id,
                    'weapon': detection['class'],
                    'status': detection.get('status', 'available'),
                    'confidence': round(detection['confidence'], 2),
                    'bbox': detection['bbox']
                })
            
            # Step 6: Annotate frame if requested
            if annotate:
                results['annotated_frame'] = self._annotate_frame(frame, stable_detections)
            
            logger.info(f"Processed frame: {len(results['detections'])} stable detections")
            return results
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {
                'detections': [],
                'tracked_objects': {},
                'frame_info': {'error': str(e)},
                'annotated_frame': frame if annotate else None
            }
    
    def cleanup(self):
        """Clean up temporary directories and resources"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


# Module-level functions
_detector_instance = None


def get_detector(yolo_model_path="models/buy_1.pt", resnet_model_path="models/buy_2.pt"):
    """Get or create detector instance (singleton pattern)"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = BuyPhaseDetector(yolo_model_path, resnet_model_path)
    return _detector_instance


def process_frame(frame, annotate=False, save_crops=False, 
                 yolo_model_path="models/buy_1.pt", resnet_model_path="models/buy_2.pt"):
    """
    Main entry point for processing a single frame
    
    Args:
        frame (numpy.ndarray): Input frame from OpenCV
        annotate (bool): Whether to return annotated frame
        save_crops (bool): Whether to save weapon crops
        yolo_model_path (str): Path to YOLO model
        resnet_model_path (str): Path to ResNet model
        
    Returns:
        dict: Detection results
    """
    detector = get_detector(yolo_model_path, resnet_model_path)
    return detector.process_frame(frame, annotate=annotate, save_crops=save_crops)


def should_run(frame_count, interval=2):
    """
    Determine if processing should run for this frame
    
    Args:
        frame_count (int): Current frame number
        interval (int): Process every N frames
        
    Returns:
        bool: True if processing should run
    """
    return frame_count % interval == 0


def reset_detector():
    """Reset the detector instance (useful for testing)"""
    global _detector_instance
    if _detector_instance is not None:
        _detector_instance.cleanup()
        _detector_instance = None


def write_results_to_file(results, output_file="buy_phase_state.json"):
    """
    Write detection results to JSON file
    
    Args:
        results (dict): Detection results from process_frame
        output_file (str): Output file path
    """
    try:
        output_data = {
            'detections': results.get('detections', []),
            'frame_info': results.get('frame_info', {}),
            'timestamp': cv2.getTickCount()
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.debug(f"Results written to {output_file}")
        
    except Exception as e:
        logger.error(f"Error writing results to file: {e}")


# Example usage
if __name__ == "__main__":
    # This should not be run when imported as a module
    logger.info("Buy Phase Detection Module loaded successfully")
    logger.info("Use process_frame() to process individual frames")

#key modules to be imported: process_frame, should_run, write_results_to_file, reset_detector    