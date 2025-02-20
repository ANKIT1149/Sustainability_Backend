import os
import zipfile
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from PIL import Image
from tqdm import tqdm
import io
import json
from collections import Counter
import gdown

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_ID = (
    "https://drive.google.com/file/d/1Y_F7MrMsqLrTGP74Drs3sSeHa0desjqZ/view?usp=sharing"
)
RESNET_ID = (
    "https://drive.google.com/file/d/1Cdt-khi_26zzAPhrPzIFFF9J6tiO40pN/view?usp=sharing"
)

# Model File & Class Labels File
MODEL_FILE = "best_waste_classifier.pth"
CLASSES_FILE = "class_labels.json"
RESNET_WEIGHTS = "resnet50_pretrained.pth"


def download_file(file_id, file_name):
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(file_name):
        print(f"📥 Downloading {file_name}...")
        gdown.download(url, file_name, quiet=False)
        print(f"✅ Downloaded {file_name}")


# Download missing model files
download_file(MODEL_ID, MODEL_FILE)
download_file(RESNET_ID, RESNET_WEIGHTS)

with open(CLASSES_FILE, "r") as f:
    class_labels = json.load(f)

# Load Model
model = models.resnet50(weights=None)
if os.path.exists(RESNET_WEIGHTS):
    print(f"✅ Loading pre-trained ResNet50 weights from {RESNET_WEIGHTS}...")
    model.load_state_dict(torch.load(RESNET_WEIGHTS, map_location="cpu"))
else:
    print("⚠ Pre-trained ResNet50 weights not found!")

model.fc = torch.nn.Linear(model.fc.in_features, 30)

if os.path.exists(MODEL_FILE):
    print(f"✅ Loading trained model from {MODEL_FILE}...")
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    dataset_loaded = False
else:
    print("⚠ No trained model found. Dataset required for training.")
    dataset_loaded = True 

model.to(device)

if os.path.exists(CLASSES_FILE):
    with open(CLASSES_FILE, "r") as f:
        class_labels = json.load(f)
    print("✅ Class labels loaded from file.")
else:
    class_labels = []

# Data Transformations (Needed for Prediction)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Dataset Loading (Only when training)
if dataset_loaded:
    file_id = "13Db44IjMJFXZTMOroJkNHHL4SkgGBYYF"
    destination = "dataset.zip"
    extract_folder = "Dataset/images/images"

    if not os.path.exists(extract_folder):
        print("📥 Dataset not found. Downloading...")

        gdown.download(
            f"https://drive.google.com/uc?id={file_id}", destination, quiet=False
        )

        with zipfile.ZipFile(destination, "r") as zip_ref:
            zip_ref.extractall("Dataset")

        os.remove(destination)
    else:
        print("✅ Dataset already exists. Skipping download.")

    dataset_Path = os.path.join(os.getcwd(), extract_folder)
    print(f"📂 Dataset Path: {dataset_Path}")

    fulldataset = datasets.ImageFolder(root=dataset_Path, transform=transform)
    class_labels = fulldataset.classes  # Get class names
    num_classes = len(class_labels)

    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Update model output

    print(f"Found classes: {class_labels}")
    print(f"Total images in dataset: {len(fulldataset)}")

    # Save class labels for inference (avoids dataset dependency)
    with open(CLASSES_FILE, "w") as f:
        json.dump(class_labels, f)
    print("✅ Class labels saved!")


# Prediction Function (Does NOT require dataset)
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    model.eval()

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    class_name = class_labels[predicted.item()]

    # Define recyclable classes
    recyclable_classes = {
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
    }

    return {
        "waste_type": class_name,
        "recyclable": class_name in recyclable_classes,
        "confidence": round(confidence.item() * 100, 2),
    }


if __name__ == "__main__":
    if dataset_loaded:
        print("Starting training...")

        train_size = int(0.8 * len(fulldataset))
        val_size = len(fulldataset) - train_size
        train_dataset, val_dataset = random_split(fulldataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False, num_workers=4
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        num_epochs = 10
        best_accuracy = 0.0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=f"{running_loss/len(train_loader):.4f}")

            print(f"Epoch {epoch + 1}, Loss: {running_loss/len(train_loader):.4f}")

            model.eval()
            correct, total = 0, 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total if total > 0 else 0
            print(f"Validation Accuracy: {accuracy:.2f}%")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), MODEL_FILE)
                print("✅ Model saved!")

        print("🎉 Training Complete!")
    else:
        print("✅ Model ready for inference.")
