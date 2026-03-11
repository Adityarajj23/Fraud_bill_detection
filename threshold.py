import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score

# CONFIG 
CSV_PATH = "results/ocr_test_results.csv"  # path to your pipeline output
true_label_col = "true_label"
score_col = "combined_fraud_score"

#  LOAD DATA
df = pd.read_csv(CSV_PATH)

thresholds = np.arange(0.3, 0.71, 0.01)  # sweep from 0.30 to 0.70
results = []

for t in thresholds:
    preds = np.where(df[score_col] > t, "fake", "real")
    fake_prec = precision_score(df[true_label_col], preds, pos_label="fake")
    fake_rec = recall_score(df[true_label_col], preds, pos_label="fake")
    real_prec = precision_score(df[true_label_col], preds, pos_label="real")
    real_rec = recall_score(df[true_label_col], preds, pos_label="real")
    results.append([t, fake_prec, fake_rec, real_prec, real_rec])

# FIND BEST BALANCE
res_df = pd.DataFrame(results, columns=["threshold", "fake_prec", "fake_rec", "real_prec", "real_rec"])
res_df["balance_score"] = (res_df["fake_rec"] + res_df["real_rec"]) / 2
best_row = res_df.loc[res_df["balance_score"].idxmax()]

print("\n=== Threshold Sweep Results ===")
print(res_df.to_string(index=False, formatters={"threshold": "{:.2f}".format}))
print("\nBest balance at threshold {:.2f}:".format(best_row["threshold"]))
print("Fake recall: {:.3f}, Real recall: {:.3f}".format(best_row["fake_rec"], best_row["real_rec"]))
