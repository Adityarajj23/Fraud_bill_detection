# Fraud Bill Detection (EfficientNet + OCR)

## 📌 Overview
This project is a **hybrid fraud-bill detector** that combines:
- An **image classifier** based on **EfficientNet‑B0** (fine‑tuned for binary classification: *real* vs *fake* receipts).
- An **OCR-based semantic validator** that checks consistency of billing fields (amount, GST number, transaction ID, date).
- A **fusion mechanism** that merges both signals into a final fraud score.

The goal is to detect manipulated or counterfeit receipts by leveraging both **visual anomalies** (logos, fonts, textures) and **textual inconsistencies**.

---

## 📊 Dataset
- **Stage‑1 Pretraining**: [CG1050 Dataset](https://www.kaggle.com/datasets/cg1050) — generic image classification dataset used for transfer learning.
- **Stage‑2 Fine‑tuning**: [Synthetic Receipts Dataset](https://www.kaggle.com/datasets/ankurzing/synthetic-receipts-dataset) — authentic receipts used for training and validation.
- **Synthetic Fake Receipts**: Generated using `tamper_with_ocr.py`, which edits OCR‑detected fields (amounts, GST numbers, dates) in real receipts to simulate fraud.

---

## 📂 Folder Structure
```
Fraud_bill_detection/
│
├── dataset/
│   ├── train/{real,fake}
│   ├── val/{real,fake}
│   ├── test/{real,fake}
│
├── dataset_cg1050/
│   └── train/{real,fake}   # Stage-1 pretraining dataset
│
├── models/
│   ├── fraud_detector.pth
│   └── classes.json        # {"fake": 0, "real": 1}
│
├── results/
│   └── ocr_test_results.csv
│
├── scripts/
│   ├── two_stage_train.py          # Current training script
│   ├── two_stage_train_old.py      # Earlier training variant
│   ├── patched_pipeline.py         # Offline evaluation (image+OCR fusion)
│   ├── test_pipeline_with_metrics_old.py # Image-only evaluation
│   ├── threshold.py                # Threshold tuning
│   ├── tamper_with_ocr.py          # Generates fake bills by editing OCR fields
│   ├── split_val.py                # Splits train into validation
│
├── tampering_log.csv               # Log of edits for synthetic fake receipts
├── tampering_log_test.csv          # Log for test tampering run
│
├── results/
│   └── ocr_test_results.csv
```

---

## ⚙️ How the Model Works
1. **Image Branch**
   - EfficientNet‑B0 backbone (`torchvision.models.efficientnet_b0`).
   - Classifier head replaced with a 2‑class linear layer.
   - Outputs `img_fake_score = P(fake)`.

2. **OCR Branch**
   - Extracts text fields using OCR.
   - Regex checks for billing consistency (amount, GST, txn ID, date).
   - Produces `ocr_score` (higher when fields are consistent).

3. **Fusion**
   - Final score computed as:
     ```
     combined_fraud_score = 0.7 * img_fake_score + 0.3 * (1 - ocr_score)
     ```
   - Predicts **fake** if score > threshold (default: 0.5).

---

## 🚀 Workflow
1. **Data Preparation**
   - Collect authentic receipts.
   - Generate fake receipts using `tamper_with_ocr.py`.
   - Split train/val/test using `split_val.py`.

2. **Training**
   - Run `two_stage_train.py`:
     - Stage‑1: Pretrain on CG1050.
     - Stage‑2: Fine‑tune on receipts dataset.
     - Last 3 EfficientNet blocks unfrozen.
     - Class weights adjusted to favor fake recall.

3. **Evaluation**
   - Run `patched_pipeline.py` for image+OCR fusion.
   - Outputs classification report + confusion matrix.
   - Detailed results saved in `results/ocr_test_results.csv`.

4. **Threshold Tuning**
   - Run `threshold.py` to sweep thresholds and optimize recall vs precision.

---

## 📈 Key Features
- **Hybrid Detection**: Combines visual + textual signals.
- **EfficientNet Backbone**: Lightweight, accurate, suitable for deployment.
- **Synthetic Tampering**: Realistic fake receipts for balanced training.
- **Explainability**: OCR logs + fraud score breakdown for transparency.

---

## 🛠️ Installation
```bash
git clone https://github.com/Adityarajj23/Fraud_bill_detection.git
cd Fraud_bill_detection
pip install -r requirements.txt
```
## ▶️ Usage 
---

Train the model
```python scripts/two_stage_train.py```

Evaluate with image + OCR fusion
```python scripts/patched_pipeline.py```

Tune threshold
```python scripts/threshold.py```

Generate synthetic fake receipts
```python scripts/tamper_with_ocr.py```

## Next Steps
---
Add Grad‑CAM visualizations for highlighting manipulated regions.

Expand dataset with multilingual receipts for better generalization.

Benchmark against ResNet and other CNN baselines.


