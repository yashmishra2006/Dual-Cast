"""
DualCast - Game Phase Detection Module
=====================================

This module handles event detection and kill feed analysis during Valorant's game phase.
Uses YOLO for event HUD detection and ResNet for agent classification in kill feeds.

Dependencies:
- ultralytics (YOLO)
- torch, torchvision (ResNet)
- opencv-python (cv2)
- PIL (Pillow)

Models required:
- yolo_game.pt (YOLO model for event detection)
- resnet_killfeed.pt (ResNet model for agent classification)
"""

import json
import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentClassifier(nn.Module):
    """ResNet-based agent classifier for kill feed analysis"""
    
    def __init__(self, num_classes=21):  # Valorant has 21+ agents
        super(AgentClassifier, self).__init__()
        # Load pretrained ResNet-18
        from torchvision.models import resnet18
        self.backbone = resnet18(pretrained=False)
        # Replace final layer for agent classification
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)


class GamePhaseDetector:
    """Main class for game phase event detection and kill feed analysis"""
    
    def __init__(self, yolo_model_path="yolo_game.pt", resnet_model_path="resnet_killfeed.pt"):
        """
        Initialize the game phase detector
        
        Args:
            yolo_model_path (str): Path to YOLO model file
            resnet_model_path (str): Path to ResNet model file
        """
        self.yolo_model_path = yolo_model_path
        self.resnet_model_path = resnet_model_path
        self.output_file = "game_phase_state.json"
        
        # Model placeholders
        self.yolo_model = None
        self.resnet_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Event class mapping (based on your YOLO model classes)
        self.event_classes = {
            0: "ace",
            1: "spike_diffusing",
            2: "ally_spike",
            3: "clutch",
            4: "enemy_kill",
            5: "flawless",
            6: "lost",
            7: "scoreboard",
            8: "spike_initiating",
            9: "spike_planted",
            10: "team_kill",
            11: "won"
        }
        
        # Kill feed event types
        self.kill_feed_events = {"enemy_kill", "team_kill"}
        
        # Agent class mapping (Valorant agents)
        self.agent_classes = {
            0: "Breach", 1: "Brimstone", 2: "Cypher", 3: "Jett", 4: "Omen",
            5: "Phoenix", 6: "Raze", 7: "Reyna", 8: "Sage", 9: "Sova",
            10: "Viper", 11: "Killjoy", 12: "Skye", 13: "Yoru", 14: "Astra",
            15: "KAY/O", 16: "Chamber", 17: "Neon", 18: "Fade", 19: "Harbor",
            20: "Gekko"
        }
        
        # Image preprocessing for ResNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Detection confidence threshold
        self.confidence_threshold = 0.5
        
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
                self.resnet_model = AgentClassifier(num_classes=len(self.agent_classes))
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
    
    def _detect_events(self, frame):
        """
        Detect game events using YOLO
        
        Args:
            frame (numpy.ndarray): Input frame from OpenCV
            
        Returns:
            dict: Dictionary with events and kill feed detections
        """
        if self.yolo_model is None:
            logger.warning("YOLO model not loaded")
            return {"events": [], "kill_feed_boxes": []}
        
        try:
            # Run YOLO detection
            results = self.yolo_model(frame)
            
            events = []
            kill_feed_boxes = []
            
            # Process detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract class, coordinates and confidence
                        cls = int(box.cls[0].cpu().numpy())
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Filter by confidence threshold
                        if confidence < self.confidence_threshold:
                            continue
                        
                        # Get event name
                        event_name = self.event_classes.get(cls, "unknown")
                        
                        # Check if it's a kill feed event
                        if event_name in self.kill_feed_events:
                            kill_feed_boxes.append({
                                'type': event_name,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence)
                            })
                        else:
                            # Regular event
                            events.append(event_name)
            
            return {
                "events": list(set(events)),  # Remove duplicates
                "kill_feed_boxes": kill_feed_boxes
            }
            
        except Exception as e:
            logger.error(f"Error in event detection: {e}")
            return {"events": [], "kill_feed_boxes": []}
    
    def _split_kill_feed(self, kill_feed_crop):
        """
        Split kill feed crop into 3 equal segments for agent classification
        
        Args:
            kill_feed_crop (numpy.ndarray): Cropped kill feed image
            
        Returns:
            list: List of 3 agent icon segments
        """
        height, width = kill_feed_crop.shape[:2]
        segment_width = width // 3
        
        segments = []
        for i in range(3):
            start_x = i * segment_width
            end_x = start_x + segment_width
            segment = kill_feed_crop[:, start_x:end_x]
            segments.append(segment)
        
        return segments
    
    def _classify_agent(self, agent_crop):
        """
        Classify agent using ResNet
        
        Args:
            agent_crop (numpy.ndarray): Cropped agent icon image
            
        Returns:
            tuple: (agent_name, confidence)
        """
        if self.resnet_model is None:
            logger.warning("ResNet model not loaded")
            return "Unknown", 0.0
        
        try:
            # Skip if crop is too small
            if agent_crop.shape[0] < 10 or agent_crop.shape[1] < 10:
                return "Unknown", 0.0
            
            # Convert BGR to RGB
            agent_crop_rgb = cv2.cvtColor(agent_crop, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(agent_crop_rgb)
            
            # Apply transforms
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.resnet_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # Get predicted class and confidence
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item()
                
                agent_name = self.agent_classes.get(predicted_class, "Unknown")
                
                return agent_name, confidence
                
        except Exception as e:
            logger.error(f"Error in agent classification: {e}")
            return "Unknown", 0.0
    
    def _process_kill_feed(self, frame, kill_feed_boxes):
        """
        Process kill feed detections to extract agent information
        
        Args:
            frame (numpy.ndarray): Input frame
            kill_feed_boxes (list): List of kill feed bounding boxes
            
        Returns:
            list: List of kill feed entries with agent information
        """
        kill_feed_results = []
        
        for kill_feed in kill_feed_boxes:
            kill_type = kill_feed['type']
            bbox = kill_feed['bbox']
            
            # Crop kill feed from frame
            x1, y1, x2, y2 = bbox
            kill_feed_crop = frame[y1:y2, x1:x2]
            
            # Skip if crop is too small
            if kill_feed_crop.shape[0] < 20 or kill_feed_crop.shape[1] < 60:
                continue
            
            # Split kill feed into 3 segments
            segments = self._split_kill_feed(kill_feed_crop)
            
            # Classify agents in each segment
            agent_predictions = []
            for segment in segments:
                agent_name, confidence = self._classify_agent(segment)
                if confidence > 0.3:  # Only include confident predictions
                    agent_predictions.append((agent_name, confidence))
            
            # Use the most confident agent prediction
            if agent_predictions:
                best_agent = max(agent_predictions, key=lambda x: x[1])
                kill_feed_results.append({
                    "type": kill_type,
                    "agent": best_agent[0]
                })
        
        return kill_feed_results
    
    def _write_output(self, results):
        """
        Write detection results to JSON file
        
        Args:
            results (dict): Detection results dictionary
        """
        try:
            with open(self.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.debug(f"Results written to {self.output_file}")
        except Exception as e:
            logger.error(f"Error writing output file: {e}")
    
    def process_game_frame(self, frame):
        """
        Process a single frame for game phase detection
        
        Args:
            frame (numpy.ndarray): Input frame from OpenCV
            
        Returns:
            dict: Game phase detection results
        """
        try:
            # Step 1: Detect events using YOLO
            detection_results = self._detect_events(frame)
            
            # Step 2: Process kill feed if detected
            kill_feed_results = []
            if detection_results["kill_feed_boxes"]:
                kill_feed_results = self._process_kill_feed(frame, detection_results["kill_feed_boxes"])
            
            # Step 3: Compile final results
            results = {
                "events": detection_results["events"],
                "kill_feed": kill_feed_results
            }
            
            # Step 4: Write results to JSON file
            self._write_output(results)
            
            logger.info(f"Processed frame: {len(results['events'])} events, {len(results['kill_feed'])} kill feeds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing game frame: {e}")
            # Write empty results on error
            empty_results = {"events": [], "kill_feed": []}
            self._write_output(empty_results)
            return empty_results


# Global detector instance
_detector = None


def _get_detector():
    """Get or create detector instance"""
    global _detector
    if _detector is None:
        _detector = GamePhaseDetector()
    return _detector


def should_run(frame_count):
    """
    Determine if processing should run for this frame
    
    Args:
        frame_count (int): Current frame number
        
    Returns:
        bool: True if processing should run
    """
    # Run every 3 frames to balance performance and responsiveness
    return frame_count % 3 == 0


def process_game_frame(frame):
    """
    Main entry point for game phase processing
    
    Args:
        frame (numpy.ndarray): Input frame from OpenCV
        
    Returns:
        dict: Game phase detection results
    """
    detector = _get_detector()
    return detector.process_game_frame(frame)


# Example usage and testing
if __name__ == "__main__":
    # Test the module
    import numpy as np
    
    # Create a dummy frame for testing
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Test should_run function
    print("Testing should_run function:")
    for i in range(10):
        print(f"Frame {i}: {should_run(i)}")
    
    # Test process_game_frame (will handle missing models gracefully)
    print("\nTesting process_game_frame:")
    results = process_game_frame(test_frame)
    print(f"Results: {results}")
    
    # Check if output file was created
    if os.path.exists("game_phase_state.json"):
        with open("game_phase_state.json", 'r') as f:
            file_contents = json.load(f)
        print(f"Output file contents: {file_contents}")
    
    # Test example output format
    example_output = {
        "events": ["spike_planted", "clutch"],
        "kill_feed": [
            {"type": "enemy_kill", "agent": "Phoenix"},
            {"type": "team_kill", "agent": "Sova"}
        ]
    }
    print(f"\nExample output format: {json.dumps(example_output, indent=2)}")