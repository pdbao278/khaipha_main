"""
TOP 5 CAU HINH - FULL 11 FEATURES (KHONG MI)
Uu tien: Recall > 77% (hon thi cang tot), Precision va F1 khong qua thap
Chien luoc: Extreme Recall (PR-Curve based) va Balance F1 
"""

import pandas as pd
import numpy as np
import warnings
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve
)

warnings.filterwarnings('ignore')

def p(msg):
    print(msg, flush=True)

# ===========================================================
# 1. DOC DU LIEU
# ===========================================================
p("=" * 80)
p("DANG DOC DU LIEU...")
p("=" * 80)

train_df = pd.read_csv('train_resampled.csv')
val_df = pd.read_csv('val.csv')
test_df = pd.read_csv('test.csv')

X_train = train_df.drop('Stroke', axis=1)
y_train = train_df['Stroke']
X_val = val_df.drop('Stroke', axis=1)
y_val = val_df['Stroke']
X_test = test_df.drop('Stroke', axis=1)
y_test = test_df['Stroke']

feature_names = X_train.columns.tolist()

p(f"  Tap Train (SMOTE): {X_train.shape[0]} mau, {X_train.shape[1]} features")
p(f"  Tap Validation:    {X_val.shape[0]} mau")
p(f"  Tap Test:          {X_test.shape[0]} mau")
p(f"  Features:          {feature_names}")
p(f"  Phan phoi Train: {y_train.value_counts().to_dict()}")
p(f"  Phan phoi Val:   {y_val.value_counts().to_dict()}")
p(f"  Phan phoi Test:  {y_test.value_counts().to_dict()}")

# ===========================================================
# 2. HAM TIM NGUONG TOI UU TREN VALIDATION
# ===========================================================
def find_extreme_recall_threshold(y_true, y_proba, target_recall=0.80):
    """
    Chien luoc Extreme Recall: Tim nguong cho Recall dat tren target (83%)
    ma F1 dat tot nhat co the. Day la chien luoc dung trong mohinh.md
    
    Logic: 
    1. Duyet tat ca nguong tu 0.01 -> 0.99
    2. Voi moi nguong, tinh Recall va F1 tren Validation
    3. Loc cac nguong dat Recall >= target_recall
    4. Trong so do, chon nguong co F1 cao nhat
    """
    best_threshold = 0.5
    best_f1 = -1
    best_recall = 0
    
    # Duyet tu nguong cao xuong thap (de tim nguong cao nhat dat recall target)
    for t in np.arange(0.01, 0.99, 0.0005):
        y_pred = (y_proba >= t).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Chi xet cac nguong dat Recall >= target
        if rec >= target_recall:
            # Trong so cac nguong dat recall, chon cai co F1 tot nhat
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
                best_recall = rec
    
    # Neu khong nguong nao dat target recall, chon nguong cho recall cao nhat
    if best_f1 == -1:
        best_recall_val = 0
        for t in np.arange(0.01, 0.99, 0.0005):
            y_pred = (y_proba >= t).astype(int)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            # Tim nguong cho recall cao nhat va F1 khong te qua
            score = rec * 0.7 + f1 * 0.3
            if score > best_f1:
                best_f1 = score
                best_threshold = t
                best_recall = rec
    
    return best_threshold

def get_proba(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))

def evaluate(model, X, y, threshold):
    y_proba = get_proba(model, X)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        'recall': recall_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'auc': roc_auc_score(y, y_proba),
        'threshold': threshold
    }

# ===========================================================
# 3. CAU HINH TOP 5 MOI THUAT TOAN
# ===========================================================

lr_configs = [
    {'name': 'LR-C1', 'params': {'C': 0.01, 'penalty': 'l2', 'class_weight': 'balanced', 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}},
    {'name': 'LR-C2', 'params': {'C': 0.1,  'penalty': 'l2', 'class_weight': 'balanced', 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}},
    {'name': 'LR-C3', 'params': {'C': 1.0,  'penalty': 'l2', 'class_weight': 'balanced', 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}},
    {'name': 'LR-C4', 'params': {'C': 10.0, 'penalty': 'l2', 'class_weight': 'balanced', 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}},
    {'name': 'LR-C5', 'params': {'C': 100.0,'penalty': 'l2', 'class_weight': 'balanced', 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}},
]

svm_configs = [
    {'name': 'SVM-C1', 'params': {'C': 0.1,   'kernel': 'rbf', 'gamma': 'scale', 'class_weight': 'balanced', 'probability': True, 'random_state': 42, 'cache_size': 1000}},
    {'name': 'SVM-C2', 'params': {'C': 1.0,   'kernel': 'rbf', 'gamma': 'scale', 'class_weight': 'balanced', 'probability': True, 'random_state': 42, 'cache_size': 1000}},
    {'name': 'SVM-C3', 'params': {'C': 10.0,  'kernel': 'rbf', 'gamma': 0.01,    'class_weight': 'balanced', 'probability': True, 'random_state': 42, 'cache_size': 1000}},
    {'name': 'SVM-C4', 'params': {'C': 10.0,  'kernel': 'rbf', 'gamma': 0.1,     'class_weight': 'balanced', 'probability': True, 'random_state': 42, 'cache_size': 1000}},
    {'name': 'SVM-C5', 'params': {'C': 100.0, 'kernel': 'rbf', 'gamma': 0.01,    'class_weight': 'balanced', 'probability': True, 'random_state': 42, 'cache_size': 1000}},
]

rf_configs = [
    {'name': 'RF-C1', 'params': {'max_depth': 4,  'n_estimators': 50,  'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}},
    {'name': 'RF-C2', 'params': {'max_depth': 6,  'n_estimators': 100, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}},
    {'name': 'RF-C3', 'params': {'max_depth': 8,  'n_estimators': 200, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}},
    {'name': 'RF-C4', 'params': {'max_depth': 8,  'n_estimators': 300, 'min_samples_split': 10, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}},
    {'name': 'RF-C5', 'params': {'max_depth': 12, 'n_estimators': 500, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}},
]

xgb_configs = [
    {'name': 'XGB-C1', 'params': {'max_depth': 3,  'learning_rate': 0.1,  'n_estimators': 100, 'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1}},
    {'name': 'XGB-C2', 'params': {'max_depth': 5,  'learning_rate': 0.05, 'n_estimators': 200, 'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1}},
    {'name': 'XGB-C3', 'params': {'max_depth': 4,  'learning_rate': 0.1,  'n_estimators': 150, 'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1}},
    {'name': 'XGB-C4', 'params': {'max_depth': 6,  'learning_rate': 0.1,  'n_estimators': 300, 'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1}},
    {'name': 'XGB-C5', 'params': {'max_depth': 10, 'learning_rate': 0.01, 'n_estimators': 500, 'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1}},
]

dt_configs = [
    {'name': 'DT-C1', 'params': {'max_depth': 3,  'min_samples_split': 2,  'class_weight': 'balanced', 'random_state': 42}},
    {'name': 'DT-C2', 'params': {'max_depth': 5,  'min_samples_split': 5,  'class_weight': 'balanced', 'random_state': 42}},
    {'name': 'DT-C3', 'params': {'max_depth': 6,  'min_samples_leaf': 20,  'class_weight': 'balanced', 'random_state': 42}},
    {'name': 'DT-C4', 'params': {'max_depth': 8,  'min_samples_split': 10, 'class_weight': 'balanced', 'random_state': 42}},
    {'name': 'DT-C5', 'params': {'max_depth': 12, 'min_samples_split': 20, 'class_weight': 'balanced', 'random_state': 42}},
]

# ===========================================================
# 4. CHAY VA DANH GIA
# ===========================================================

all_results = {}

algorithms = [
    ('Logistic Regression', LogisticRegression, lr_configs),
    ('Random Forest', RandomForestClassifier, rf_configs),
    ('XGBoost', XGBClassifier, xgb_configs),
    ('Decision Tree', DecisionTreeClassifier, dt_configs),
    ('SVM (RBF)', SVC, svm_configs),
]

for algo_name, ModelClass, configs in algorithms:
    p(f"\n{'=' * 80}")
    p(f"  THUAT TOAN: {algo_name}")
    p(f"{'=' * 80}")
    
    algo_results = []
    
    for cfg in configs:
        cfg_name = cfg['name']
        params = cfg['params']
        
        p(f"\n  [{cfg_name}] Huan luyen...")
        
        start = time.time()
        model = ModelClass(**params)
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        p(f"    Xong: {train_time:.2f}s. Tim nguong...")
        
        # Tim nguong Extreme Recall tren Validation (target recall >= 83%)
        val_proba = get_proba(model, X_val)
        threshold = find_extreme_recall_threshold(y_val, val_proba, target_recall=0.83)
        
        # Danh gia tren Test
        result = evaluate(model, X_test, y_test, threshold)
        result['config_name'] = cfg_name
        result['params'] = str(params)
        result['train_time'] = train_time
        
        algo_results.append(result)
        
        mark = "OK" if result['recall'] >= 0.80 else "LOW"
        p(f"    -> Nguong={result['threshold']:.4f} | Recall={result['recall']*100:.2f}% | Prec={result['precision']*100:.2f}% | F1={result['f1']*100:.2f}% | AUC={result['auc']*100:.2f}% [{mark}]")
    
    # Sap xep: uu tien Recall >= 80% va F1 cao
    algo_results.sort(key=lambda x: (
        1 if x['recall'] >= 0.80 else 0,
        x['f1'],       # F1 cao nhat trong nhom dat recall
        x['recall'],
    ), reverse=True)
    
    all_results[algo_name] = algo_results
    
    p(f"\n  --- BANG XEP HANG TOP 5 ({algo_name}) ---")
    p(f"  {'Hang':<6} {'Config':<10} {'Recall':>8} {'Precision':>10} {'F1-Score':>10} {'AUC-ROC':>10} {'Nguong':>8}")
    p(f"  {'-'*66}")
    for i, r in enumerate(algo_results, 1):
        mark = " *" if r['recall'] >= 0.80 else ""
        p(f"  Top {i:<2} {r['config_name']:<10} {r['recall']*100:>7.2f}% {r['precision']*100:>9.2f}% {r['f1']*100:>9.2f}% {r['auc']*100:>9.2f}% {r['threshold']:>7.4f}{mark}")

# ===========================================================
# 5. TONG HOP KET QUA
# ===========================================================
p(f"\n\n{'#' * 80}")
p(f"#{'TONG HOP - TOP 5 CAU HINH/THUAT TOAN (FULL 11 FEATURES, KHONG MI)':^78}#")
p(f"{'#' * 80}")

p(f"\n{'='*105}")
p(f"{'Thuat toan':<25} {'Config':<10} {'Recall':>8} {'Prec':>8} {'F1':>8} {'AUC':>8} {'Nguong':>8} {'Time':>10}")
p(f"{'='*105}")

for algo_name, results in all_results.items():
    for i, r in enumerate(results):
        algo_display = algo_name if i == 0 else ""
        mark = " *" if r['recall'] >= 0.80 else "  "
        p(f"{algo_display:<25} {r['config_name']:<10} {r['recall']*100:>7.2f}% {r['precision']*100:>7.2f}% {r['f1']*100:>7.2f}% {r['auc']*100:>7.2f}% {r['threshold']:>7.4f} {r['train_time']:>8.2f}s{mark}")
    p(f"{'-'*105}")

# ===========================================================
# 6. LUU BAO CAO
# ===========================================================
report = []
report.append("=" * 90)
report.append("BAO CAO TOP 5 CAU HINH MOI THUAT TOAN - FULL 11 FEATURES (KHONG MI)")
report.append(f"Ngay chay: {time.strftime('%Y-%m-%d %H:%M:%S')}")
report.append("Chien luoc nguong: Extreme Recall (target Recall >= 83% tren Validation)")
report.append("Uu tien: Recall > 80% tren Test, F1 va Precision khong qua thap")
report.append("=" * 90)
report.append("")
report.append(f"TAP DU LIEU:")
report.append(f"  - Train (sau SMOTE 0.5): {X_train.shape[0]} mau")
report.append(f"  - Validation: {X_val.shape[0]} mau")
report.append(f"  - Test: {X_test.shape[0]} mau")
report.append(f"  - So features: {X_train.shape[1]}")
report.append(f"  - Features: {feature_names}")
report.append(f"  - Phan chia: 70% Train / 10% Val / 20% Test")
report.append("")

for algo_name, results in all_results.items():
    report.append("=" * 90)
    report.append(f"THUAT TOAN: {algo_name}")
    report.append("=" * 90)
    
    for i, r in enumerate(results, 1):
        status = "DAT (>= 80%)" if r['recall'] >= 0.80 else "CHUA DAT (< 80%)"
        report.append(f"\n  --- Top {i}: {r['config_name']} ---")
        report.append(f"  Sieu tham so: {r['params']}")
        report.append(f"  Chien luoc nguong: Extreme Recall")
        report.append(f"  Nguong quyet dinh: {r['threshold']:.4f}")
        report.append(f"  Thoi gian huan luyen: {r['train_time']:.2f}s")
        report.append(f"  KET QUA TREN TAP TEST:")
        report.append(f"    - Recall (Lop 1):    {r['recall']*100:.2f}%  [{status}]")
        report.append(f"    - Precision (Lop 1): {r['precision']*100:.2f}%")
        report.append(f"    - F1-Score (Lop 1):  {r['f1']*100:.2f}%")
        report.append(f"    - AUC-ROC:           {r['auc']*100:.2f}%")
    report.append("")

report.append("\n" + "=" * 90)
report.append("BANG TONG HOP CAU HINH TOT NHAT MOI THUAT TOAN (Top 1)")
report.append("=" * 90)
report.append(f"{'Thuat toan':<25} {'Config':<10} {'Recall':>10} {'Prec':>10} {'F1':>10} {'AUC':>10} {'Nguong':>8}")
report.append("-" * 90)

for algo_name, results in all_results.items():
    r = results[0]
    report.append(f"{algo_name:<25} {r['config_name']:<10} {r['recall']*100:>9.2f}% {r['precision']*100:>9.2f}% {r['f1']*100:>9.2f}% {r['auc']*100:>9.2f}% {r['threshold']:>7.4f}")

with open('report_top5_full11.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(report))

p(f"\n>>> Bao cao da luu vao: report_top5_full11.txt")
p(f">>> HOAN TAT!")
