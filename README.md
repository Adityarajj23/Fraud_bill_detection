# Fraud Bill Detection

## Folder Structure

```plaintext
/Fraud Bill Detection
 ├── data/
 │   ├── raw/
 │   ├── processed/
 ├── models/
 ├── notebooks/
 ├── src/
 │   ├── __init__.py
 │   ├── data_preprocessing.py
 │   ├── model_training.py
 │   └── detection.py
 └── README.md
```

## Model Execution Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Adityarajj23/Fraud_bill_detection.git
   cd Fraud_bill_detection
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Preprocess the data:
   ```bash
   python src/data_preprocessing.py
   ```
4. Train the model:
   ```bash
   python src/model_training.py
   ```
5. Run the detection:
   ```bash
   python src/detection.py
   ```

## Detection Details
The model employs a combination of machine learning algorithms to identify fraudulent transactions. It uses a dataset of historical transactions to learn patterns indicative of fraud.