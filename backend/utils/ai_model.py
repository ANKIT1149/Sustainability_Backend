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
from collections import Counter
import gdown

# model training code

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

file_id = "13Db44IjMJFXZTMOroJkNHHL4SkgGBYYF"
destination = "dataset.zip"
extract_folder = "Dataset/images/images"

if not os.path.exists(extract_folder):
    print("📥 Dataset not found. Downloading...")

    path = gdown.download(
    f"https://drive.google.com/uc?id={file_id}", destination, quiet=False
    )

    with zipfile.ZipFile(destination, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    os.remove(destination)

else:
    print("✅ Dataset already exists. Skipping download.")


dataset_Path = os.path.join(os.getcwd(), extract_folder)
print(f"📂 Dataset Path: {dataset_Path}")


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

fulldataset = datasets.ImageFolder(root=dataset_Path, transform=transform)
num_classes = len(fulldataset.classes)
print(f"Found classes: {fulldataset.classes}")
print(f"Total images in dataset: {len(fulldataset)}")


class_counts = Counter([label for _, label in fulldataset.samples])
print(
    "Class Distribution:", {fulldataset.classes[k]: v for k, v in class_counts.items()}
)

class_weights = torch.tensor(
    [1.0 / class_counts[i] for i in range(num_classes)], dtype=torch.float
).to(device)
sampler = WeightedRandomSampler(
    [class_weights[label] for _, label in fulldataset.samples],
    num_samples=len(fulldataset),
    replacement=True,
)


MODEL_FILE = "best_waste_classifier.pth"
model = models.resnet50(weights="IMAGENET1K_V2")
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.to(device)

if os.path.exists(MODEL_FILE):
    print(f"Loading trained model from {MODEL_FILE}...")
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
else:
    print("No trained model found. Train the model first.")


def predict_image(image_path):
    image = Image.open(io.BytesIO(image_path)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    class_name = fulldataset.classes[predicted.item()]
    print(f"Predicted Class: {class_name}")

    print(f"Available classases: {fulldataset.classes}")

    return {
        "waste_type": class_name,
        "recyclable": class_name
        in [
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
        ],
        "confidence": round(confidence.item() * 100, 2),
    }


if __name__ == "__main__":
    print("Starting training...")

    train_size = int(0.8 * len(fulldataset))
    val_size = len(fulldataset) - train_size
    train_dataset, val_dataset = random_split(fulldataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=16, sampler=sampler, num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        correct_per_class = {classname: 0 for classname in fulldataset.classes}
        total_per_class = {classname: 0 for classname in fulldataset.classes}

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                for label, pred in zip(labels, predicted):
                    class_name = fulldataset.classes[label.item()]
                    total_per_class[class_name] += 1
                    if label == pred:
                        correct_per_class[class_name] += 1

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total if total > 0 else 0
        print(f"Validation Accuracy: {accuracy:.2f}%")

        for classname in fulldataset.classes:
            acc = (
                100 * correct_per_class[classname] / total_per_class[classname]
                if total_per_class[classname] > 0
                else 0
            )
            print(f"Accuracy for {classname}: {acc:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), MODEL_FILE)
            print("✅ Model saved!")

    print("🎉 Training Complete!")
