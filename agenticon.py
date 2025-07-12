import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import argparse
import os
from PIL import Image
import json

class ValorantAgentClassifier:
    def __init__(self, model_path, num_classes=21):
        """Initialize the Valorant agent classifier"""
        print("Loading custom Valorant agent detection model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.num_classes = num_classes
        
        # Load custom model
        self.model = self.load_custom_model(model_path)
        self.model.eval()
        self.model.to(self.device)
        
        # Define image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Valorant agent class labels
        self.agent_labels = [
            "Astra", "Breach", "Brimstone", "Chamber", "Cypher",
            "Fade", "Harbor", "Jett", "KAYO", "Killjoy",
            "Neon", "Omen", "Phoenix", "Raze", "Reyna",
            "Sage", "Skye", "Sova", "Viper", "Yoru", "Gekko"
        ]
        
        # Agent colors for visualization (BGR format)
        self.agent_colors = {
            "Astra": (255, 100, 150),
            "Breach": (0, 165, 255),
            "Brimstone": (0, 100, 200),
            "Chamber": (200, 200, 0),
            "Cypher": (100, 100, 255),
            "Fade": (150, 50, 200),
            "Harbor": (255, 150, 0),
            "Jett": (255, 255, 255),
            "KAYO": (128, 128, 128),
            "Killjoy": (0, 255, 255),
            "Neon": (255, 0, 255),
            "Omen": (50, 0, 100),
            "Phoenix": (0, 140, 255),
            "Raze": (0, 0, 255),
            "Reyna": (128, 0, 128),
            "Sage": (0, 255, 0),
            "Skye": (0, 200, 100),
            "Sova": (0, 255, 200),
            "Viper": (0, 128, 0),
            "Yoru": (200, 0, 200),
            "Gekko": (100, 255, 0)
        }
        
        print("Model loaded successfully!")
    
    def load_custom_model(self, model_path):
        """Load custom trained model"""
        try:
            # Create ResNet50 backbone
            model = resnet18(weights=None)
            
            # Modify final layer for Valorant agents
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            
            # Load custom weights
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict):
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"Loaded custom model from {model_path}")
            else:
                print(f"Warning: Model file {model_path} not found. Using random weights.")
                print("Please provide a valid model path.")
            
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback: create model with random weights
            model = resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            return model
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        try:
            # Load image using PIL
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            input_tensor = self.transform(image)
            input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
            
            return input_batch.to(self.device)
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image_path):
        """Make prediction on image"""
        # Preprocess image
        input_batch = self.preprocess_image(image_path)
        if input_batch is None:
            return None
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top 5 predictions
            top5_prob, top5_indices = torch.topk(probabilities, min(5, len(self.agent_labels)))
            
            predictions = []
            for i in range(len(top5_indices)):
                idx = top5_indices[i].item()
                prob = top5_prob[i].item()
                
                # Get agent label
                if idx < len(self.agent_labels):
                    label = self.agent_labels[idx]
                else:
                    label = f"Unknown Agent {idx}"
                
                predictions.append({
                    'agent': label,
                    'confidence': prob,
                    'index': idx
                })
            
            return predictions
    
    def draw_predictions_on_image(self, image, predictions):
        """Draw prediction results on the image with Valorant styling"""
        # Create a copy of the image
        result_image = image.copy()
        h, w = result_image.shape[:2]
        
        # Create overlay for semi-transparent background
        overlay = result_image.copy()
        
        # Calculate overlay dimensions
        overlay_height = min(320, h // 2)
        overlay_width = min(450, w // 2)
        
        # Position overlay at top-left
        x_offset = 20
        y_offset = 20
        
        # Draw semi-transparent background
        cv2.rectangle(overlay, 
                     (x_offset - 10, y_offset - 10),
                     (x_offset + overlay_width, y_offset + overlay_height),
                     (0, 0, 0), -1)
        
        # Apply transparency
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, result_image, 1 - alpha, 0, result_image)
        
        # Draw title
        title_font = cv2.FONT_HERSHEY_DUPLEX
        title_text = "VALORANT AGENT DETECTION"
        cv2.putText(result_image, title_text, 
                   (x_offset, y_offset + 25), title_font, 0.7, (255, 255, 255), 2)
        
        # Draw separator line
        cv2.line(result_image, 
                (x_offset, y_offset + 35), 
                (x_offset + overlay_width - 20, y_offset + 35),
                (255, 255, 255), 1)
        
        # Draw predictions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        for i, pred in enumerate(predictions):
            y_pos = y_offset + 70 + i * 45
            
            # Get agent color
            agent_color = self.agent_colors.get(pred['agent'], (255, 255, 255))
            
            # Draw agent name
            agent_text = f"{i+1}. {pred['agent']}"
            cv2.putText(result_image, agent_text, 
                       (x_offset + 10, y_pos), font, font_scale, agent_color, thickness)
            
            # Draw confidence percentage
            confidence_text = f"{pred['confidence']:.1%}"
            cv2.putText(result_image, confidence_text, 
                       (x_offset + 200, y_pos), font, font_scale, (255, 255, 255), thickness)
            
            # Draw confidence bar
            bar_width = 150
            bar_height = 8
            bar_x = x_offset + 270
            bar_y = y_pos - 10
            
            # Background bar
            cv2.rectangle(result_image, 
                         (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height),
                         (50, 50, 50), -1)
            
            # Confidence bar
            fill_width = int(bar_width * pred['confidence'])
            cv2.rectangle(result_image, 
                         (bar_x, bar_y), 
                         (bar_x + fill_width, bar_y + bar_height),
                         agent_color, -1)
        
        # Draw model info at bottom
        info_y = y_offset + overlay_height - 20
        cv2.putText(result_image, f"Model: Custom ResNet18 | Device: {self.device.type.upper()}", 
                   (x_offset, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return result_image
    
    def classify_and_display(self, image_path):
        """Classify image and display with OpenCV"""
        # Load original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        print(f"Processing image: {image_path}")
        
        # Make prediction
        predictions = self.predict(image_path)
        if predictions is None:
            print("Error: Could not make prediction")
            return
        
        # Print results to console
        print("\n" + "="*50)
        print("VALORANT AGENT DETECTION RESULTS")
        print("="*50)
        for i, pred in enumerate(predictions):
            print(f"{i+1}. {pred['agent']:<15} - {pred['confidence']:.2%}")
        print("="*50)
        
        # Draw predictions on image
        result_image = self.draw_predictions_on_image(original_image, predictions)
        
        # Resize for display if image is too large
        display_height = 800
        if result_image.shape[0] > display_height:
            aspect_ratio = result_image.shape[1] / result_image.shape[0]
            display_width = int(display_height * aspect_ratio)
            result_image = cv2.resize(result_image, (display_width, display_height))
        
        # Display image
        cv2.imshow('Valorant Agent Detection', result_image)
        
        # Wait for key press
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Valorant Agent Icon Detection')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to the trained model file (.pth)')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to the image to classify')
    parser.add_argument('--num_classes', type=int, default=21,
                       help='Number of agent classes (default: 21)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} not found")
        return
    
    # Initialize classifier
    classifier = ValorantAgentClassifier(args.model, args.num_classes)
    
    # Classify and display
    classifier.classify_and_display(args.image)

if __name__ == "__main__":
    # Example usage if running without command line args
    if len(os.sys.argv) == 1:
        print("Example usage:")
        print("python valorant_classifier.py --model path/to/model.pth --image path/to/agent_icon.jpg")
        print("\nOr modify the script to set default paths:")
        
        # Default paths for testing (modify these)
        model_path = "agent_icon.pt"  # Path to your trained model
        image_path = "test_image.png"       # Path to test image
        
        if os.path.exists(model_path) and os.path.exists(image_path):
            classifier = ValorantAgentClassifier(model_path)
            classifier.classify_and_display(image_path)
        else:
            print("Please provide valid model and image paths")
    else:
        main()