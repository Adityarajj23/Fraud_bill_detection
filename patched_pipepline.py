import os
import json
import torch
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import pytesseract
import re

# CONFIG 
MODEL_PATH = "models/fraud_detector.pth"
CLASS_MAP_PATH = "models/classes.json"
TEST_DIR = "dataset/test"
OUTPUT_CSV = "results/ocr_test_results.csv"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#  ENSURE SAVE FOLDER 
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# CHECK FILES 
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")
if not os.path.exists(CLASS_MAP_PATH):
    raise FileNotFoundError(f"❌ Class mapping file not found: {CLASS_MAP_PATH}")

#  LOAD MODEL 
model = models.efficientnet_b0()
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

with open(CLASS_MAP_PATH, "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
fake_index = [i for i, c in idx_to_class.items() if c.lower() == "fake"][0]

# TRANSFORMS 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# OCR CHECK 
def ocr_check(img: Image.Image):
    # Downscale large images
    MAX_OCR_WIDTH = 1000
    if img.width > MAX_OCR_WIDTH:
        ratio = MAX_OCR_WIDTH / img.width
        img = img.resize((MAX_OCR_WIDTH, int(img.height * ratio)))

    # Convert to grayscale
    img = img.convert("L")

    try:
        text = pytesseract.image_to_string(img, timeout=5)
    except RuntimeError:
        print("⚠️ OCR timed out for this image, skipping.")
        return 0, {"amount": [], "txn_id": [], "gst_no": [], "date": []}

    amount_pattern = r"(?i)\b(?:amount|total amount|total)\b(?:\s|\n){0,2}(?:₹|rs\.?|inr|\$|usd|eur)\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?"
    txn_pattern = r"\b(?:TXN|UID|TID)\d{6,12}\b"
    gst_pattern = r"(?i)\bGST(?:IN|(?:\s*Reg)?(?:\s*No\.?)?)?[:\s]*[A-Z0-9]{6,15}\b"
    date_pattern = r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b"

    amount_found = re.findall(amount_pattern, text)
    txn_found = re.findall(txn_pattern, text)
    gst_found = re.findall(gst_pattern, text)
    date_found = re.findall(date_pattern, text)

    score = 0
    if amount_found:
        score += 0.3
    if txn_found or gst_found:
        score += 0.4
    if date_found:
        score += 0.3

    return score, {
        "amount": amount_found,
        "txn_id": txn_found,
        "gst_no": gst_found,
        "date": date_found
    }


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

        img_path = os.path.join(folder, fname)
        img = Image.open(img_path).convert("RGB")

        # Image model prediction
        img_t = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.softmax(outputs, dim=1)
            img_fake_score = float(probs[0][fake_index])

        # OCR check
        ocr_score, ocr_data = ocr_check(img)

        # Combined fraud score
        combined_score = (0.7 * img_fake_score) + (0.3 * (1 - ocr_score))
        prediction = "fake" if combined_score > 0.5 else "real"

        y_true.append(label)
        y_pred.append(prediction)

        results.append({
            "filename": fname,
            "true_label": label,
            "predicted_label": prediction,
            "image_model_score": img_fake_score,
            "ocr_score": ocr_score,
            "combined_fraud_score": combined_score,
            "ocr_data": ocr_data
        })

# METRICS 
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# SAVE RESULTS 
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"✅ Test results saved to {OUTPUT_CSV}")
