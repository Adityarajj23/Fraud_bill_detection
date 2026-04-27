# 🧾 Fraud Bill Detection
### EfficientNet-B0 + OCR Multi-Modal Validation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EfficientNet--B0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/OCR-Tesseract-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Accuracy-79%25-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Fraud%20Recall-98%25-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/ROC%20AUC-0.938-blueviolet?style=for-the-badge"/>
</p>

> A hybrid fraud-bill detector combining **EfficientNet-B0 visual classification** and **OCR-based semantic validation** to detect manipulated or counterfeit receipts — achieving a fraud recall of **98%** and ROC AUC of **0.938**.

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Why EfficientNet-B0?](#-why-efficientnet-b0)
- [How It Works](#%EF%B8%8F-how-it-works)
- [Results](#-results)
- [Dataset](#-dataset)
- [Folder Structure](#-folder-structure)
- [Installation](#%EF%B8%8F-installation)
- [Usage](#-usage)
- [Next Steps](#-next-steps)

---

## 📌 Overview

Financial document fraud is a growing problem across banking, healthcare, retail, and enterprise expense management. This project proposes a **two-branch pipeline** that catches fraud from two independent angles:

| Branch | What it Detects |
|--------|----------------|
| 🖼️ **Image Branch** (EfficientNet-B0) | Visual anomalies — font inconsistencies, logo distortion, compression artefacts from editing |
| 📝 **OCR Branch** (Tesseract + Regex) | Semantic inconsistencies — missing/malformed amounts, GST numbers, transaction IDs, dates |

Neither branch alone is sufficient — a visually intact bill can have doctored fields, and a bill with valid fields can still show visual manipulation. **Fusion addresses both failure modes simultaneously.**

---

## 🏗️ Why EfficientNet-B0?

**EfficientNet-B0** was chosen over alternative architectures for the following reasons:

> **Primary Constraint:** This project uses a **self-curated, limited dataset** (~1,200 training samples). Forged document datasets raise serious security and legal concerns, making it impractical to source large-scale public datasets. Therefore, **the model architecture must be optimized for small-data regimes**, prioritizing efficient transfer learning over brute-force scaling.

| Criterion | EfficientNet-B0 | ResNet-50 | MobileNet-v3 | ViT-Base |
|-----------|-----------------|-----------|------------|----------|
| **Parameters** | ~5.3M | ~26M | ~5.4M | ~87M |
| **Model Size** | ~21 MB | ~98 MB | ~22 MB | ~345 MB |
| **Inference Speed (CPU)** | ⚡ Fast | ⚠️ Slow | ⚡ Fast | 🐌 Very Slow |
| **Top-1 ImageNet Accuracy** | 77.1% | 76.1% | 75.9% | 81.1% |
| **Deployment Suitability** | ✅ Excellent | ⚠️ Moderate | ✅ Excellent | ❌ Poor |
| **Small Dataset Robustness** | ✅ Excellent | ⚠️ Prone to Overfitting | ⚠️ Moderate | ❌ Poor |

### **Key Advantages:**

1. **Optimized for Small Datasets** — With only ~1,200 training images, EfficientNet-B0's 5.3M parameters minimize overfitting risk. Larger models (ResNet-50: 26M, ViT-Base: 87M) require 10,000+ images to prevent memorization on limited data.

2. **Efficient Transfer Learning** — Pre-trained on ImageNet, it rapidly adapts to fraud detection without needing massive domain-specific data:
   - Stage 1 learns generic bill features on CG1050 (quick convergence with few samples)
   - Stage 2 specializes on receipt manipulation patterns with minimal fine-tuning

3. **Optimal Accuracy-to-Speed Tradeoff** — EfficientNet-B0 achieves 77.1% ImageNet accuracy with only 5.3M parameters, making it ideal for real-time API responses without sacrificing performance.

4. **Production Deployment** — The 21 MB model size enables:
   - Fast REST API inference (~50-100ms per request on CPU)
   - Low memory footprint for horizontal scaling
   - Viable deployment on edge/mobile devices

5. **Computational Efficiency** — Achieves **98% fraud recall** and **ROC AUC = 0.938** without requiring GPU acceleration, reducing infrastructure costs.

6. **Proven on Document Classification** — EfficientNet-B0 is widely used in document analysis tasks (invoices, receipts, passports) where visual consistency detection is critical.

### **Why Not ResNet-50?**
ResNet-50's 26M parameters are excessive for a 1,200-image dataset and would lead to severe overfitting. While it offers 76.1% ImageNet accuracy, it requires 5-10× more training data to leverage its capacity. Additionally, its 98 MB model size introduces unnecessary inference latency.

### **Why Not MobileNet-v3?**
While MobileNet-v3 offers similar efficiency (5.4M params), it underperforms EfficientNet-B0 on general visual tasks and shows less robust transfer learning on domain-specific datasets like receipts — particularly critical when training data is scarce.

### **Why Not Vision Transformer (ViT)?**
ViT-Base requires 87M parameters and massive datasets (10M+ images) to train effectively. On a 1,200-image dataset, ViT would catastrophically overfit. Even with pre-training, it's computationally expensive for inference and designed for large-scale problems, making it fundamentally unsuitable for this constrained scenario.

---

## ⚙️ How It Works

### System Pipeline

```
                        Input Bill Image
                               │
                               ▼
               ┌─── Preprocessing (224×224) ───┐
               │    ImageNet Normalisation      │
               └───────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
   ┌─────────────────────┐       ┌─────────────────────────┐
   │   BRANCH A          │       │   BRANCH B              │
   │   EfficientNet-B0   │       │   OCR + Regex Validator │
   │                     │       │                         │
   │  Stage 1: Pretrain  │       │  Extract text via OCR   │
   │  Stage 2: Fine-tune │       │  Regex match:           │
   │  Unfreeze last 3    │       │   • Amount              │
   │  blocks             │       │   • GST Number          │
   │  Class-weighted     │       │   • Transaction ID      │
   │  loss               │       │   • Date                │
   │                     │       │                         │
   │  s_img ∈ [0,1]      │       │  s_ocr = matched/4      │
   └────────┬────────────┘       └────────────┬────────────┘
            │                                 │
            └──────────────┬──────────────────┘
                           ▼
               ┌───────────────────────┐
               │     Score Fusion      │
               │                       │
               │  S = 0.7 × s_img      │
               │    + 0.3 × (1−s_ocr)  │
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │  Threshold Decision   │
               │  τ = 0.5 (eval)       │
               │  τ = 0.65 (API)       │
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │  Output: REAL / FAKE  │
               │  + OCR Field Report   │
               └───────────────────────┘
```

### Two-Stage Transfer Learning

```
Stage 1 ──► Pretrain on CG1050 dataset
            (Classifier head only — learn general bill features)
                           │
                           ▼
Stage 2 ──► Fine-tune on receipts dataset
            (Unfreeze last 3 EfficientNet blocks + head)
            (Class-weighted loss to boost fake recall)
```

### Fusion Formula

```
combined_fraud_score = 0.7 × img_fake_score + 0.3 × (1 − ocr_score)

If combined_fraud_score > τ  →  FAKE
Else                         →  REAL
```

The **70/30 weighting** prioritises visual signals while letting OCR provide a meaningful correction — a bill missing all four key fields gets pushed toward fraud even if the image looks clean.

---

## 📈 Results

### Evaluation Curves

Threshold-independent performance on the held-out test set (694 samples, balanced).
The **red dot** marks the operating point at τ = 0.5.

![ROC Curve and Precision-Recall Curve](assets/curves.png)

| Metric | Value |
|--------|-------|
| 🟣 **ROC AUC** | **0.938** |
| 🟠 **PR AUC (Average Precision)** | **0.935** |
| Random Classifier Baseline | 0.500 |

> **What this means:** A ROC AUC of 0.938 confirms the model strongly separates real from fake bills at *any* threshold. A PR AUC of 0.935 vs. a 0.50 random baseline means the high false-positive rate at τ = 0.5 is a **deliberate threshold choice**, not a fundamental model weakness — shift τ upward to trade recall for precision along the PR curve.

---

### Classification Report

Evaluated at threshold τ = 0.5 on 694 test samples (347 real, 347 fake):

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Fake** | 0.71 | **0.98** ✅ | 0.82 | 347 |
| **Real** | 0.96 | 0.60 | 0.74 | 347 |
| Macro Avg | 0.84 | 0.79 | 0.78 | 694 |
| **Accuracy** | | | **0.79** | 694 |

---

### Confusion Matrix

|  | Predicted Fake | Predicted Real |
|--|:--------------:|:--------------:|
| **Actual Fake** | **339** ✅ TP | 8 ❌ FN |
| **Actual Real** | 140 ⚠️ FP | **207** ✅ TN |

---

### Interpretation

- ✅ **Fake recall = 0.98** — only 8 out of 347 fraudulent bills slip through undetected. In fraud detection, a missed fake is far costlier than a false alarm.
- ⚠️ **Real recall = 0.60** — 140 genuine bills are over-flagged. This is an intentional consequence of class-weighted loss and the 70/30 fusion weight, both designed to minimise missed fraud.
- 🎯 **ROC AUC = 0.938** — the model is inherently strong; threshold tuning via `threshold.py` can shift the operating point to suit different deployment requirements (stricter for banking, looser for retail).

---

## 📊 Dataset

| Source | Purpose | Notes |
|--------|---------|-------|
| [CG1050 Dataset](https://www.kaggle.com/datasets/cg1050) | Stage-1 pretraining | Generic image classification for transfer learning |
| [Synthetic Receipts Dataset](https://www.kaggle.com/datasets/ankurzing/synthetic-receipts-dataset) | Stage-2 fine-tuning | Authentic receipts for training and validation |
| `tamper_with_ocr.py` output | Fake bill generation | Programmatically edits amounts, GST, dates in real receipts |

| Split | Real | Fake | Total |
|-------|------|------|-------|
| Train | ~600 | ~600 | ~1200 |
| Test  | 347  | 347  | 694   |

---

## 📂 Folder Structure

```
Fraud_bill_detection/
│
├── assets
│   ├── curves.png
│   └── sample_dataset/           # Sample images for API testing
│       ├── real/                  # Sample real receipt
│       └── fake/                  # Sample fake receipt
│ 
├── dataset/
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
│
├── dataset_cg1050/
│   └── train/
│       ├── real/
│       └── fake/               # Stage-1 pretraining dataset
│
├── models/
│   ├── fraud_detector.pth      # Trained model weights
│   └── classes.json            # {"fake": 0, "real": 1}
│
├── results/
│   ├── ocr_test_results.csv    # Per-image scores and OCR fields
│   └── curves.png              # ROC + PR evaluation curves  ← add this file here
│
├── scripts/
│   ├── two_stage_train.py      # Two-stage training pipeline
│   ├── patched_pipeline.py     # Offline evaluation (image + OCR fusion)
│   ├── threshold.py            # Threshold sweep and tuning
│   ├── tamper_with_ocr.py      # Generates fake bills by editing OCR fields
│   └── split_val.py            # Splits train into validation
│
├── app.py                      # FastAPI application for REST API deployment
├── utils.py                    # Shared utilities (OCR check, fraud score calculation)
├── tampering_log.csv           # Log of edits for synthetic fake receipts
├── tampering_log_test.csv      # Log for test tampering run
├── requirements.txt
└── README.md
```

> 📌 **Important:** Place `curves.png` inside the `results/` folder so the image renders correctly in this README.

---

## 🛠️ Installation

```bash
git clone https://github.com/Adityarajj23/Fraud_bill_detection.git
cd Fraud_bill_detection
pip install -r requirements.txt
```

**Key dependencies:** `torch`, `torchvision`, `pytesseract`, `Pillow`, `scikit-learn`, `pandas`, `opencv-python`, `matplotlib`

---

## ▶️ Usage

### 1. Generate Synthetic Fake Receipts
```bash
python scripts/tamper_with_ocr.py
```
Edits OCR-detected fields (amounts, GST numbers, dates) in real receipts to produce realistic fake samples.

### 2. Prepare Train / Val Split
```bash
python scripts/split_val.py
```

### 3. Train the Model (Two-Stage)
```bash
python scripts/two_stage_train.py
```
- **Stage 1:** Pretrain on CG1050 (classifier head only)
- **Stage 2:** Fine-tune on receipts dataset (last 3 EfficientNet blocks + head unfrozen, class-weighted loss)

### 4. Evaluate with Image + OCR Fusion
```bash
python scripts/patched_pipeline.py
```
Outputs classification report, confusion matrix, and saves per-image scores to `results/ocr_test_results.csv`.

### 5. Tune Decision Threshold
```bash
python scripts/threshold.py
```
Sweeps thresholds and shows the precision/recall tradeoff — use this to find the optimal τ for your deployment context.

### 6. Deploy as REST API
```bash
uvicorn app:app --reload
```
Launches the FastAPI server for real-time fraud detection. Access the interactive API documentation at `http://127.0.0.1:8000/docs`.

---

## 🚀 REST API Deployment

### Running the API

1. **Activate the virtual environment** (if using `venv`):
   ```powershell
   & ./venv/Scripts/Activate.ps1
   ```

2. **Start the FastAPI server**:
   ```bash
   uvicorn app:app --reload
   ```
   - By default, the server runs on `http://127.0.0.1:8000`
   - `--reload` enables auto-reloading during development

3. **Access Swagger UI** (interactive API documentation):
   - Open your browser and navigate to: `http://127.0.0.1:8000/docs`
   - Or use the ReDoc alternative: `http://127.0.0.1:8000/redoc`

### API Features

#### Endpoint: `POST /predict`

**Description:** Upload an image file and get a fraud detection prediction.

**Parameters:**
- `file` (multipart/form-data): Image file to analyze (JPG, PNG, JPEG)
- `threshold` (query parameter, optional): Fraud detection threshold
  - Allowed values: `0.5` (strict, for banking) or `0.65` (default, for general use)
  - Default: `0.65`

**Response:**
```json
{
  "image_model_score": 0.85,
  "ocr_score": 0.7,
  "combined_fraud_score": 0.71,
  "prediction": "fake",
  "threshold_used": 0.65,
  "ocr_data": {
    "amount": ["₹500.00"],
    "txn_id": ["TXN123456"],
    "gst_no": ["27AABCT1234G1Z5"],
    "date": ["15-04-2024"]
  }
}
```

**Response Fields:**
- `image_model_score`: EfficientNet-B0 confidence score (0–1) for fake classification
- `ocr_score`: OCR-based validation score (0–1) based on detected fields
- `combined_fraud_score`: Fused score using the formula: `0.7 × image_model_score + 0.3 × (1 − ocr_score)`
- `prediction`: Classification result — `"real"` or `"fake"`
- `threshold_used`: The threshold applied for this prediction
- `ocr_data`: Extracted OCR fields (amount, transaction ID, GST number, date)

#### Endpoint: `GET /`

**Description:** Health check endpoint to verify the API is running.

**Response:**
```json
{
  "message": "Fraud detection API is running. Use POST /predict to send an image."
}
```

### Use Case: Threshold Selection

| Use Case | Threshold | Rationale |
|----------|-----------|-----------|
| 🏦 **Banking / Financial** | `0.5` | Strict — prioritize fraud detection over false alarms |
| 🏫 **Education / General** | `0.65` | Balanced — reasonable trade-off between precision and recall |

**Example cURL Request (Strict Banking):**
```bash
curl -X POST "http://127.0.0.1:8000/predict?threshold=0.5" \
  -F "file=@path/to/image.jpg"
```

**Example cURL Request (Default General Use):**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@path/to/image.jpg"
```

### Code Reusability

The API and offline evaluation pipeline (`patched_pipeline.py`) share common utilities through `utils.py`:
- `ocr_check()` — OCR field extraction and validation
- `calculate_fraud_score()` — Score fusion and prediction logic

This ensures consistency across both deployment modes and eliminates code duplication.

### Testing with Sample Images

Quick-start test data is included in the repository for easy API validation:

- **Location:** `assets/sample_dataset/`
  - `real/` — contains 2 sample real receipt
  - `fake/` — contains 2 sample fake receipt

**To test via Swagger UI:**
1. Navigate to `http://127.0.0.1:8000/docs`
2. Click **"Try it out"** on the `POST /predict` endpoint
3. Click **"Choose File"** and select an image from `assets/sample_dataset/real/` or `assets/sample_dataset/fake/`
4. (Optional) Adjust the `threshold` parameter (`0.5` or `0.65`)
5. Click **"Execute"** to see the prediction and OCR extraction

**Example cURL test with sample image:**
```bash
curl -X POST "http://127.0.0.1:8000/predict?threshold=0.65" \
  -F "file=@assets/sample_dataset/real/sample_real.jpg"
```

---

## 🔮 Next Steps

- [ ] **Grad-CAM visualisations** — highlight manipulated regions within flagged receipts for explainability
- [ ] **Multilingual receipts** — expand dataset to improve generalisation across languages and formats
- [ ] **Threshold auto-tuning** — integrate F-beta optimisation directly into the training loop

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
