import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import json

# ===== CONFIG =====
CG1050_TRAIN_DIR = "dataset_cg1050/train"
TXN_TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MODEL_PATH = "models/fraud_detector.pth"
CLASS_MAP_PATH = "models/classes.json"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== ENSURE SAVE FOLDERS =====
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ===== TRANSFORMS =====
train_transforms = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== DATA LOADER =====
def get_loader(data_dir, transform, shuffle=True):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Dataset path not found: {data_dir}")
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    print(f"📂 Loaded {len(dataset)} images from {data_dir} | Classes: {dataset.classes}")
    return dataset, loader

# ===== MODEL SETUP =====
def build_model():
    model = models.efficientnet_b0(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    return model.to(DEVICE)

# ===== TRAIN FUNCTION =====
def train_stage(model, train_loader, val_loader=None, epochs=3, lr=1e-3, class_weights=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_corrects += (outputs.argmax(1) == labels).sum().item()

        train_acc = running_corrects / len(train_loader.dataset)

        if val_loader:
            model.eval()
            val_corrects = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    val_corrects += (outputs.argmax(1) == labels).sum().item()
            val_acc = val_corrects / len(val_loader.dataset)
            print(f"📊 Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
        else:
            print(f"📊 Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f}")

    return model

# ===== MAIN PIPELINE =====
if __name__ == "__main__":
    # Stage 1: Pre-train on CG‑1050
    print("=== Stage 1: Pre-training on CG‑1050 ===")
    cg_dataset, cg_loader = get_loader(CG1050_TRAIN_DIR, train_transforms)
    model = build_model()
    model = train_stage(model, cg_loader, epochs=5, lr=1e-3)

    # Stage 2: Fine-tune on transaction dataset
    print("\n=== Stage 2: Fine-tuning on Transaction Dataset ===")
    txn_dataset, txn_loader = get_loader(TXN_TRAIN_DIR, train_transforms)
    val_dataset, val_loader = get_loader(VAL_DIR, val_transforms, shuffle=False)

    # Unfreeze last 3 blocks for deeper adaptation
    for param in model.features[-3:].parameters():
        param.requires_grad = True

    # Class weights to favour catching fakes (index 0 = fake, 1 = real)
    class_weights = torch.tensor([1.2, 1.0]).to(DEVICE)

    model = train_stage(model, txn_loader, val_loader, epochs=8, lr=5e-4, class_weights=class_weights)

    # Save model + class mapping
    torch.save(model.state_dict(), MODEL_PATH)
    with open(CLASS_MAP_PATH, "w") as f:
        json.dump(txn_dataset.class_to_idx, f)

    print(f"✅ Final model saved to {MODEL_PATH}")
    print(f"✅ Class mapping saved to {CLASS_MAP_PATH}")
