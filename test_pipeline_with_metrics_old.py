import os, json, torch
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# CONFIG 
MODEL_PATH = "models/fraud_detector.pth"
CLASS_MAP_PATH = "models/classes.json"
TEST_DIR = "dataset/test"
OUTPUT_CSV = "results/new_test_results.csv"

# ENSURE SAVE FOLDERS EXIST 
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# LOAD MODEL 
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")
if not os.path.exists(CLASS_MAP_PATH):
    raise FileNotFoundError(f"❌ Class mapping file not found: {CLASS_MAP_PATH}")

model = models.efficientnet_b0()
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

with open(CLASS_MAP_PATH, "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# TRANSFORMS 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# TEST LOOP 
y_true, y_pred, results = [], [], []
for label in ["real", "fake"]:
    folder = os.path.join(TEST_DIR, label)
    if not os.path.exists(folder):
        print(f"⚠️ Skipping missing folder: {folder}")
        continue
    for fname in os.listdir(folder):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = Image.open(os.path.join(folder, fname)).convert("RGB")
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
            pred_idx = probs.argmax(1).item()
        y_true.append(label)
        y_pred.append(idx_to_class[pred_idx])
        results.append({
            "filename": fname,
            "true_label": label,
            "predicted_label": idx_to_class[pred_idx],
            "fake_score": float(probs[0][class_to_idx["fake"]])
        })

#  METRICS 
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ===== SAVE RESULTS =====
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"✅ Test results saved to {OUTPUT_CSV}")
