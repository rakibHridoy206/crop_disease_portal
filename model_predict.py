import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import logging
from typing import Dict, Union, Tuple
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CropDiseasePredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = self._load_labels()
        self.model = self._load_model()
        self.transform = self._get_transforms()
        
    def _load_labels(self) -> list:
        try:
            with open("model/labels.txt", "r") as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            logging.error("Labels file not found")
            raise
            
    def _load_model(self) -> torch.nn.Module:
        try:
            model = torch.load("model/model.pth", map_location=self.device)
            model.eval()
            return model
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    def _get_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])

    def _check_image_quality(self, image: Image) -> Tuple[bool, str]:
        """Check if image meets quality standards"""
        np_image = np.array(image)
        
        # Check brightness
        brightness = np.mean(np_image)
        if brightness < 30 or brightness > 240:
            return False, "Poor image brightness"
            
        # Check contrast
        contrast = np.std(np_image)
        if contrast < 20:
            return False, "Poor image contrast"
            
        return True, "OK"

    def predict_frame(self, frame_path: str) -> Dict[str, Union[str, float]]:
        """
        Enhanced prediction with confidence scores and quality checks
        """
        try:
            # Load and check image
            image = Image.open(frame_path).convert("RGB")
            quality_ok, message = self._check_image_quality(image)
            
            if not quality_ok:
                logging.warning(f"Image quality issue: {message}")
                return {
                    "label": "Unknown",
                    "confidence": 0.0,
                    "quality_check": message
                }

            # Prepare input
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Get prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Get top 3 predictions
                top3_prob, top3_indices = torch.topk(probabilities, 3)
                top3_labels = [self.labels[idx] for idx in top3_indices[0]]
                top3_conf = [float(p) for p in top3_prob[0]]

            result = {
                "label": self.labels[predicted.item()],
                "confidence": float(confidence.item()),
                "quality_check": "Pass",
                "top3_predictions": list(zip(top3_labels, top3_conf)),
                "disease_severity": self._analyze_severity(image)
            }

            logging.info(f"Prediction complete: {result['label']} ({result['confidence']:.2f})")
            return result

        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            return {
                "label": "Error",
                "confidence": 0.0,
                "error": str(e)
            }

    def _analyze_severity(self, image: Image) -> Dict[str, Union[float, str]]:
        """Analyze disease severity based on image characteristics"""
        np_image = np.array(image)
        hsv = cv2.cvtColor(np_image, cv2.COLOR_RGB2HSV)
        
        # Define healthy and diseased color ranges
        healthy_mask = cv2.inRange(hsv, 
                                 np.array([35, 50, 50]), 
                                 np.array([85, 255, 255]))
        
        diseased_mask = cv2.inRange(hsv, 
                                  np.array([0, 50, 50]), 
                                  np.array([35, 255, 255]))
        
        total_pixels = np_image.shape[0] * np_image.shape[1]
        diseased_ratio = cv2.countNonZero(diseased_mask) / total_pixels
        
        severity = diseased_ratio * 100
        return {
            "percentage": float(severity),
            "level": "High" if severity > 50 else "Medium" if severity > 25 else "Low"
        }