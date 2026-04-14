import pytesseract
import re
from PIL import Image

# OCR Check Function
def ocr_check(img: Image.Image):
    text = pytesseract.image_to_string(img)
    amount_pattern = r"(?i)\b(?:amount|total amount|total)\b(?:\s|\n){0,2}(?:₹|rs\.?|inr|\$|usd|eur)\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?"
    txn_pattern = r"\b(?:TXN|UID|TID)\d{6,12}\b"
    gst_pattern = r"(?i)\bGST(?:IN|(?:\s*Reg)?(?:\s*No\.?)?)?[:\s]*[A-Z0-9]{6,15}(?=[A-Z0-9]*\d)"
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

# Fraud Score Calculation
def calculate_fraud_score(img_fake_score, ocr_score, threshold=0.65):
    combined_score = (0.7 * img_fake_score) + (0.3 * (1 - ocr_score))
    prediction = "fake" if combined_score > threshold else "real"
    return combined_score, prediction