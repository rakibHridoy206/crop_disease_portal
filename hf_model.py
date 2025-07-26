from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch

class CropDiseaseViT:
    def __init__(self):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("wambugu71/crop_leaf_diseases_vit")
        self.model = ViTForImageClassification.from_pretrained("wambugu71/crop_leaf_diseases_vit")
        self.labels = self.model.config.id2label

    def predict(self, frame_path: str):
        image = Image.open(frame_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]
        score, idx = torch.max(probs, dim=0)
        return {"label": self.labels[idx.item()], "confidence": score.item()}
