import torch
from torchvision import transforms
from PIL import Image
import os

# Load labels
LABELS = []
with open("model/labels.txt", "r") as f:
    LABELS = [line.strip() for line in f.readlines()]

# Load model
MODEL_PATH = "model/model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

def predict_frame_real(frame_path: str) -> str:
    """
    Run real prediction on frame using loaded PyTorch model.
    """
    image = Image.open(frame_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label_idx = predicted.item()
        label = LABELS[label_idx]

    return label