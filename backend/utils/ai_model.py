import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

MODEL_FILE = "best_waste_classifier.pth"
num_classes = 30  # Update based on your dataset

model = models.resnet50(weights="IMAGENET1K_V2")
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

if torch.cuda.is_available():
    model.load_state_dict(torch.load(MODEL_FILE))
else:
    model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device("cpu")))

model.eval()
print("✅ Model Loaded for Deployment!")

class_labels = [
    "plastic_cup_lids",
    "plastic_detergent_bottles",
    "plastic_food_containers",
    "plastic_shopping_bags",
    "plastic_soda_bottles",
    "plastic_straws",
    "plastic_trash_bags",
    "plastic_water_bottles",
    "newspaper",
    "magazines",
    "office_paper",
    "cardboard_boxes",
    "cardboard_packaging",
    "glass_beverage_bottles",
    "glass_cosmetic_containers",
    "glass_food_jars",
    "aluminum_food_cans",
    "aluminum_soda_cans",
    "steel_food_cans",
]


def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    class_name = class_labels[predicted.item()]
    recyclable_classes = set(class_labels)

    return {
        "waste_type": class_name,
        "recyclable": class_name in recyclable_classes,
        "confidence": round(confidence.item() * 100, 2),
    }


print("🚀 Model Ready for Inference!")
