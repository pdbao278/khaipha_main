import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Load data
train_df = pd.read_csv("train_mi_top5.csv")
val_df = pd.read_csv("val_mi_top5.csv")
test_df = pd.read_csv("test_mi_top5.csv")
TARGET_COL = "Stroke"

features = [col for col in train_df.columns if col != TARGET_COL]
x_train = train_df[features]
y_train = train_df[TARGET_COL]
x_val = val_df[features]
y_val = val_df[TARGET_COL]
x_test = test_df[features]
y_test = test_df[TARGET_COL]

# SVM-C5 model setup
model = SVC(
    C=100.0, 
    kernel='rbf', 
    gamma=0.01, 
    class_weight='balanced', 
    probability=True, 
    random_state=42
)
print("Training SVM-C5...")
model.fit(x_train, y_train)
print("Finished training.")

# Get probabilities
val_proba = model.predict_proba(x_val)[:, 1]
test_proba = model.predict_proba(x_test)[:, 1]

# 1. Kich ban 1: Fixed threshold = 0.1765
threshold_fixed = 0.1765
test_pred_fixed = (test_proba >= threshold_fixed).astype(int)

rec_fixed = recall_score(y_test, test_pred_fixed)
prec_fixed = precision_score(y_test, test_pred_fixed)
f1_fixed = f1_score(y_test, test_pred_fixed)
auc_fixed = roc_auc_score(y_test, test_proba)

print("\n--- SCENARIO 1: FIXED THRESHOLD (0.1765) ---")
print(f"Recall:    {rec_fixed*100:.4f}%")
print(f"Precision: {prec_fixed*100:.4f}%")
print(f"F1-Score:  {f1_fixed*100:.4f}%")
print(f"AUC-ROC:   {auc_fixed*100:.4f}%")

# 2. Kich ban 2: Dynamic threshold on Val (target recall >= 0.83)
best_threshold = 0.5
best_f1 = -1
for threshold in np.arange(0.01, 0.99, 0.0005):
    y_pred = (val_proba >= threshold).astype(int)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    if rec >= 0.83 and f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nOptimal dynamic threshold found on Val: {best_threshold:.4f}")

test_pred_dynamic = (test_proba >= best_threshold).astype(int)
rec_dyn = recall_score(y_test, test_pred_dynamic)
prec_dyn = precision_score(y_test, test_pred_dynamic)
f1_dyn = f1_score(y_test, test_pred_dynamic)

print("\n--- SCENARIO 2: DYNAMIC THRESHOLD ---")
print(f"Recall:    {rec_dyn*100:.4f}%")
print(f"Precision: {prec_dyn*100:.4f}%")
print(f"F1-Score:  {f1_dyn*100:.4f}%")
print(f"AUC-ROC:   {auc_fixed*100:.4f}%")

# Verify details on test set predictions
tn, fp, fn, tp = pd.crosstab(y_test, test_pred_dynamic).values.flatten()
print(f"\nConfusion Matrix (Dynamic):")
print(f"TN: {tn}, FP: {fp}")
print(f"FN: {fn}, TP: {tp}")
print(f"Calculated Recall: {tp / (tp + fn) * 100:.4f}%")
print(f"Calculated Precision: {tp / (tp + fp) * 100:.4f}%")
