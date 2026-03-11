import os, random, csv, re
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import pytesseract

REAL_DIR = "dataset/test/real"
FAKE_DIR = "dataset/test/fake"
LOG_FILE = "tampering_log_test.csv"
os.makedirs(FAKE_DIR, exist_ok=True)
FONT_PATH = "arial.ttf"

def random_txn_id(): return f"TXN{random.randint(100000, 999999)}"
def random_amount(): return f"₹{random.randint(100, 9999)}.{random.randint(0,99):02d}"
def random_date():
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 9, 1)
    delta = end_date - start_date
    return (start_date + timedelta(days=random.randint(0, delta.days))).strftime("%d-%m-%Y")

def tamper_image_with_ocr(image_path, save_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, size=24)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    changes = {"txn_id": ("", ""), "amount": ("", ""), "date": ("", "")}

    for i, word in enumerate(data["text"]):
        if not word.strip(): continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if re.match(r"TXN\d{6,}", word, re.IGNORECASE):
            new_val = random_txn_id(); changes["txn_id"] = (word, new_val)
        elif re.match(r"₹?\d+(?:\.\d{1,2})?", word):
            new_val = random_amount(); changes["amount"] = (word, new_val)
        elif re.match(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", word):
            new_val = random_date(); changes["date"] = (word, new_val)
        else: continue
        draw.rectangle([x, y, x+w, y+h], fill="white")
        draw.text((x, y), new_val, font=font, fill=(0, 0, 0))

    if random.random() > 0.5:
        width, height = img.size
        x1, y1 = random.randint(0, width//2), random.randint(0, height//2)
        x2, y2 = x1 + random.randint(50, 150), y1 + random.randint(20, 80)
        region = img.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(radius=2))
        img.paste(region, (x1, y1, x2, y2))

    img.save(save_path)
    return changes

with open(LOG_FILE, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=[
        "original_file", "tampered_file",
        "original_txn_id", "new_txn_id",
        "original_amount", "new_amount",
        "original_date", "new_date"
    ])
    writer.writeheader()
    for fname in os.listdir(REAL_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")): continue
        fake_name = f"tampered_{fname}"
        changes = tamper_image_with_ocr(os.path.join(REAL_DIR, fname), os.path.join(FAKE_DIR, fake_name))
        writer.writerow({
            "original_file": fname, "tampered_file": fake_name,
            "original_txn_id": changes["txn_id"][0], "new_txn_id": changes["txn_id"][1],
            "original_amount": changes["amount"][0], "new_amount": changes["amount"][1],
            "original_date": changes["date"][0], "new_date": changes["date"][1]
        })

print(f"✅ Tampered images saved to {FAKE_DIR}")
print(f"✅ Tampering log saved to {LOG_FILE}")
