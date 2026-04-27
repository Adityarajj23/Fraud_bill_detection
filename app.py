from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from PIL import Image
import torch
from torchvision import models, transforms
import pytesseract
import io
import re
import json
import os
from utils import ocr_check, calculate_fraud_score

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

# ===== API ENDPOINT =====
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Query(0.65, description="Fraud detection threshold (0.5 or 0.65)")
):
    # Restrict threshold to 0.5 or 0.65
    if threshold not in [0.5, 0.65]:
        raise HTTPException(status_code=400, detail="Threshold must be either 0.5 (strict) or 0.65 (default)")

    img = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Image model prediction
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        img_fake_score = float(probs[0][fake_index])

    # OCR Check
    ocr_score, ocr_data = ocr_check(img)

    # Fraud Score Calculation
    combined_score, prediction = calculate_fraud_score(img_fake_score, ocr_score, threshold)

    return {
        "image_model_score": img_fake_score,
        "ocr_score": ocr_score,
        "combined_fraud_score": combined_score,
        "prediction": prediction,
        "threshold_used": threshold,
        "ocr_data": ocr_data
    }

@app.get("/")
def read_root():
    return {"message": "Fraud detection API is running. Use POST /predict to send an image."}
