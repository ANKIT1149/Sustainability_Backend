import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from collections import Counter
from tqdm import tqdm

# ✅ Detect Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Dataset Path
dataset_Path = os.path.join(os.getcwd(), "dataset", "images", "images")

# ✅ Define Image Transformations
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# ✅ Load Dataset
fulldataset = datasets.ImageFolder(root=dataset_Path, transform=transform)
num_classes = len(fulldataset.classes)
print(f"Found classes: {fulldataset.classes}")
print(f"Total images in dataset: {len(fulldataset)}")

# ✅ Handle Class Imbalance (Weighted Sampling)
class_counts = Counter([label for _, label in fulldataset.samples])
class_weights = torch.tensor(
    [1.0 / class_counts[i] for i in range(num_classes)], dtype=torch.float
).to(device)
sampler = WeightedRandomSampler(
    [class_weights[label] for _, label in fulldataset.samples],
    num_samples=len(fulldataset),
    replacement=True,
)

# ✅ Split Dataset
train_size = int(0.8 * len(fulldataset))
val_size = len(fulldataset) - train_size
train_dataset, val_dataset = random_split(fulldataset, [train_size, val_size])

# ✅ DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# ✅ Load Pretrained Model
model = models.resnet50(weights="IMAGENET1K_V2")
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# ✅ Loss Function & Optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ✅ Train Model
num_epochs = 10
best_accuracy = 0.0
MODEL_FILE = "best_waste_classifier.pth"

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

    # ✅ Validation
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

    # ✅ Save Best Model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), MODEL_FILE)
        print("✅ Model saved!")

print("🎉 Training Complete! Model Ready for Deployment 🚀")
