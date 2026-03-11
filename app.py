from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import models, transforms
import pytesseract
import io
import re
import json
import os

# ===== CONFIG =====
MODEL_PATH = "old_models/fraud_detector_old.pth"
CLASS_MAP_PATH = "old_models/classes_old.json"
# Explicit Tesseract path for venv safety
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ===== CHECK MODEL FILES =====
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")
if not os.path.exists(CLASS_MAP_PATH):
    raise FileNotFoundError(f"❌ Class mapping file not found: {CLASS_MAP_PATH}")

# ===== APP =====
app = FastAPI()

# ===== LOAD MODEL =====
model = models.efficientnet_b0()
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ===== LOAD CLASS MAPPING =====
with open(CLASS_MAP_PATH, "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
fake_index = [i for i, c in idx_to_class.items() if c.lower() == "fake"][0]

# ===== PREPROCESSING =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== OCR CHECK =====
def ocr_check(img: Image.Image):
    text = pytesseract.image_to_string(img)

    # Amount: keywords + currency + number
    amount_pattern = r"(?i)\b(?:amount|total amount|total)\b(?:\s|\n){0,2}(?:₹|rs\.?|inr|\$|usd|eur)\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?"

    # Transaction ID: TXN, UID, or TID followed by 6–12 digits
    txn_pattern = r"\b(?:TXN|UID|TID)\d{6,12}\b"

    # GST number: GSTIN or GST with 6–15 alphanumeric chars
    gst_pattern = r"(?i)\bGST(?:IN|(?:\s*Reg)?(?:\s*No\.?)?)?[:\s]*[A-Z0-9]{6,15}(?=[A-Z0-9]*\d)"

    # Date: dd-mm-yyyy, dd/mm/yyyy, yyyy-mm-dd, dd Mon yyyy
    date_pattern = r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b"

    # Find matches
    amount_found = re.findall(amount_pattern, text)
    txn_found = re.findall(txn_pattern, text)
    gst_found = re.findall(gst_pattern, text)
    date_found = re.findall(date_pattern, text)

    # Score calculation
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

# ===== API ENDPOINT =====
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")

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
    THRESHOLD = 0.65  # tuned for best balance
    prediction = "fake" if combined_score > THRESHOLD else "real"

    return {
        "image_model_score": img_fake_score,
        "ocr_score": ocr_score,
        "combined_fraud_score": combined_score,
        "prediction": prediction,
        "ocr_data": ocr_data
    }

@app.get("/")
def read_root():
    return {"message": "Fraud detection API is running. Use POST /predict to send an image."}
