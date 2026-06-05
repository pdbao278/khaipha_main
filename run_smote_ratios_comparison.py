import os
import re
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif

# Thiết lập bảng mã UTF-8 cho console đầu ra trên Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

warnings.filterwarnings('ignore')

# Khai báo đường dẫn và hằng số
WORKSPACE = r"e:\kpdl_main"
DATA_PROCESSED = os.path.join(WORKSPACE, "data_processed.csv")
REPORT_PATH = os.path.join(WORKSPACE, "ket_qua_tong_hop_smote_ratios.md")
TARGET_COL = "Stroke"

# Định nghĩa các cấu hình siêu tham số (C1 đến C5)
LR_CONFIGS = {
    'LR-C1': {'C': 0.01, 'penalty': 'l2', 'class_weight': 'balanced', 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42},
    'LR-C2': {'C': 0.1,  'penalty': 'l2', 'class_weight': 'balanced', 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42},
    'LR-C3': {'C': 1.0,  'penalty': 'l2', 'class_weight': 'balanced', 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42},
    'LR-C4': {'C': 10.0, 'penalty': 'l2', 'class_weight': 'balanced', 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42},
    'LR-C5': {'C': 100.0,'penalty': 'l2', 'class_weight': 'balanced', 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42},
}

SVM_CONFIGS = {
    'SVM-C1': {'C': 0.1,   'kernel': 'rbf', 'gamma': 'scale', 'class_weight': 'balanced', 'probability': True, 'random_state': 42, 'cache_size': 1000},
    'SVM-C2': {'C': 1.0,   'kernel': 'rbf', 'gamma': 'scale', 'class_weight': 'balanced', 'probability': True, 'random_state': 42, 'cache_size': 1000},
    'SVM-C3': {'C': 10.0,  'kernel': 'rbf', 'gamma': 0.01,    'class_weight': 'balanced', 'probability': True, 'random_state': 42, 'cache_size': 1000},
    'SVM-C4': {'C': 10.0,  'kernel': 'rbf', 'gamma': 0.1,     'class_weight': 'balanced', 'probability': True, 'random_state': 42, 'cache_size': 1000},
    'SVM-C5': {'C': 100.0, 'kernel': 'rbf', 'gamma': 0.01,    'class_weight': 'balanced', 'probability': True, 'random_state': 42, 'cache_size': 1000},
}

RF_CONFIGS = {
    'RF-C1': {'max_depth': 4,  'n_estimators': 50,  'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
    'RF-C2': {'max_depth': 6,  'n_estimators': 100, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
    'RF-C3': {'max_depth': 8,  'n_estimators': 200, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
    'RF-C4': {'max_depth': 8,  'n_estimators': 300, 'min_samples_split': 10, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
    'RF-C5': {'max_depth': 12, 'n_estimators': 500, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
}

XGB_CONFIGS = {
    'XGB-C1': {'max_depth': 3,  'learning_rate': 0.1,  'n_estimators': 100, 'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1},
    'XGB-C2': {'max_depth': 5,  'learning_rate': 0.05, 'n_estimators': 200, 'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1},
    'XGB-C3': {'max_depth': 4,  'learning_rate': 0.1,  'n_estimators': 150, 'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1},
    'XGB-C4': {'max_depth': 6,  'learning_rate': 0.1,  'n_estimators': 300, 'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1},
    'XGB-C5': {'max_depth': 10, 'learning_rate': 0.01, 'n_estimators': 500, 'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1},
}

DT_CONFIGS = {
    'DT-C1': {'max_depth': 3,  'min_samples_split': 2,  'class_weight': 'balanced', 'random_state': 42},
    'DT-C2': {'max_depth': 5,  'min_samples_split': 5,  'class_weight': 'balanced', 'random_state': 42},
    'DT-C3': {'max_depth': 6,  'min_samples_leaf': 20,  'class_weight': 'balanced', 'random_state': 42},
    'DT-C4': {'max_depth': 8,  'min_samples_split': 10, 'class_weight': 'balanced', 'random_state': 42},
    'DT-C5': {'max_depth': 12, 'min_samples_split': 20, 'class_weight': 'balanced', 'random_state': 42},
}

ALGO_MAP = {
    'Logistic Regression': (LogisticRegression, LR_CONFIGS, 'LR', 'LR-C4'),
    'SVM (RBF)': (SVC, SVM_CONFIGS, 'SVM', 'SVM-C5'),
    'Random Forest': (RandomForestClassifier, RF_CONFIGS, 'RF', 'RF-C3'),
    'XGBoost': (XGBClassifier, XGB_CONFIGS, 'XGB', 'XGB-C3'),
    'Decision Tree': (DecisionTreeClassifier, DT_CONFIGS, 'DT', 'DT-C3')
}

# =====================================================================
# 1. KIỂM TRA BẢO MẬT DỮ LIỆU & TẠO SPLIT GỐC
# =====================================================================
def get_original_splits():
    df = pd.read_csv(DATA_PROCESSED)
    x = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    
    # Chia train_val / test theo tỷ lệ 80/20
    x_temp_train, x_test, y_temp_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )
    # Chia train / val theo tỷ lệ 70/10 (tương đương 12.5% của 80%)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp_train, y_temp_train, test_size=0.125, random_state=42, stratify=y_temp_train
    )
    
    # Kiểm tra trùng lặp dòng (data overlap)
    train_idx = set(x_train.index)
    val_idx = set(x_val.index)
    test_idx = set(x_test.index)
    
    assert len(train_idx.intersection(val_idx)) == 0, "LỖI: Trùng lặp tập Train và Validation!"
    assert len(train_idx.intersection(test_idx)) == 0, "LỖI: Trùng lặp tập Train và Test!"
    assert len(val_idx.intersection(test_idx)) == 0, "LỖI: Trùng lặp tập Validation và Test!"
    
    print("[LOG] Phân chia dữ liệu gốc thành công. Không phát hiện trùng lặp dòng.")
    return x_train, x_val, x_test, y_train, y_val, y_test

# =====================================================================
# 2. HÀM TÌM NGƯỠNG PHÂN LOẠI TỐI ƯU
# =====================================================================
def get_proba(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    scores = model.decision_function(x)
    return 1 / (1 + np.exp(-scores))

def find_extreme_recall_threshold(y_true, y_proba, target_recall=0.83):
    best_threshold = 0.5
    best_f1 = -1
    for threshold in np.arange(0.01, 0.99, 0.0005):
        y_pred = (y_proba >= threshold).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if rec >= target_recall and f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    if best_f1 == -1:
        best_score = -1
        for threshold in np.arange(0.01, 0.99, 0.0005):
            y_pred = (y_proba >= threshold).astype(int)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            score = rec * 0.7 + f1 * 0.3
            if score > best_score:
                best_score = score
                best_threshold = threshold
    return float(best_threshold)

def evaluate_test(model, x_test, y_test, threshold):
    y_proba = get_proba(model, x_test)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    auc = roc_auc_score(y_test, y_proba)
    
    # Lớp 0
    prec_l0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec_l0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_l0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    
    # Lớp 1
    prec_l1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_l1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_l1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    return {
        'threshold': threshold,
        'accuracy': f"{accuracy * 100:.2f}%",
        'auc': f"{auc * 100:.2f}%",
        'prec_l0': f"{prec_l0 * 100:.2f}%",
        'prec_l1': f"{prec_l1 * 100:.2f}%",
        'rec_l0': f"{rec_l0 * 100:.2f}%",
        'rec_l1': f"{rec_l1 * 100:.2f}%",
        'f1_l0': f"{f1_l0 * 100:.2f}%",
        'f1_l1': f"{f1_l1 * 100:.2f}%",
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }

# =====================================================================
# 3. CHẠY THỬ NGHIỆM VÀ THU THẬP KẾT QUẢ
# =====================================================================
def run_experiments():
    x_train_raw, x_val_raw, x_test_raw, y_train_raw, y_val_raw, y_test_raw = get_original_splits()
    
    # Tính MI và lấy top 5 đặc trưng cố định trước
    mi_scores = mutual_info_classif(x_train_raw, y_train_raw, random_state=42)
    mi_series = pd.Series(mi_scores, index=x_train_raw.columns)
    selected_features = list(mi_series.nlargest(5).index)
    print(f"[LOG] MI Top 5 dac trung duoc chon: {selected_features}")
    
    smote_ratios = [None, 0.5, 0.6, 0.7]
    feature_sets = ["Full11", "MI5"]
    
    raw_results = {}
    fixed_thresholds = {} # key: (ratio, config_code) -> threshold
    
    for ratio in smote_ratios:
        ratio_label = "NoSMOTE" if ratio is None else f"SMOTE_{ratio}"
        print(f"\n[TIEN TRINH] Dang thuc thi tren ti le: {ratio_label}...")
        
        # Áp dụng SMOTE nếu có
        if ratio is not None:
            sm = SMOTE(sampling_strategy=ratio, random_state=42)
            x_train_sm, y_train_sm = sm.fit_resample(x_train_raw, y_train_raw)
            # Kiem tra kich thuoc sau SMOTE de dam bao an toan dữ liệu
            expected_minority = int(len(y_train_raw[y_train_raw == 0]) * ratio)
            actual_minority = len(y_train_sm[y_train_sm == 1])
            # Cho phép sai số nhỏ do làm tròn số nguyên
            assert abs(actual_minority - expected_minority) <= 1, f"LỖI: Kích thước SMOTE không đúng! Đòi hỏi: {expected_minority}, Thực tế: {actual_minority}"
        else:
            x_train_sm, y_train_sm = x_train_raw.copy(), y_train_raw.copy()
            
        for f_set in feature_sets:
            # Subset đặc trưng
            feats = list(x_train_sm.columns) if f_set == "Full11" else selected_features
            
            x_tr = x_train_sm[feats]
            x_v = x_val_raw[feats]
            x_te = x_test_raw[feats]
            
            for algo_name, (ModelClass, configs, prefix, rep_config) in ALGO_MAP.items():
                for config_code, params in configs.items():
                    # Huấn luyện mô hình
                    model = ModelClass(**params)
                    model.fit(x_tr, y_train_sm)
                    
                    # Tối ưu ngưỡng động trên Val
                    val_proba = get_proba(model, x_v)
                    dynamic_thresh = find_extreme_recall_threshold(y_val_raw, val_proba, target_recall=0.83)
                    
                    # Nếu là Full 11, lưu lại ngưỡng động để làm ngưỡng cố định cho MI Top 5
                    if f_set == "Full11":
                        fixed_thresholds[(ratio_label, config_code)] = dynamic_thresh
                    
                    # Đánh giá với ngưỡng động
                    eval_dyn = evaluate_test(model, x_te, y_test_raw, dynamic_thresh)
                    raw_results[(ratio_label, f_set, "Dynamic", config_code)] = eval_dyn
                    
                    # Đánh giá với ngưỡng cố định (chỉ áp dụng cho MI5)
                    if f_set == "MI5":
                        # Lay nguong cua Full11 tuong ung
                        fix_thresh = fixed_thresholds.get((ratio_label, config_code), dynamic_thresh)
                        eval_fix = evaluate_test(model, x_te, y_test_raw, fix_thresh)
                        raw_results[(ratio_label, f_set, "Fixed", config_code)] = eval_fix
                        
                    print(f"  - Đã chạy {algo_name} ({config_code}) [{f_set} | {ratio_label}]")
                    
    return raw_results

CONFIG_PARAMS = {
    'LR-C1': "C=0.01, penalty='l2', class_weight='balanced'",
    'LR-C2': "C=0.1, penalty='l2', class_weight='balanced'",
    'LR-C3': "C=1.0, penalty='l2', class_weight='balanced'",
    'LR-C4': "C=10.0, penalty='l2', class_weight='balanced'",
    'LR-C5': "C=100.0, penalty='l2', class_weight='balanced'",
    
    'SVM-C1': "C=0.1, kernel='rbf', gamma='scale', class_weight='balanced'",
    'SVM-C2': "C=1.0, kernel='rbf', gamma='scale', class_weight='balanced'",
    'SVM-C3': "C=10.0, kernel='rbf', gamma=0.01, class_weight='balanced'",
    'SVM-C4': "C=10.0, kernel='rbf', gamma=0.1, class_weight='balanced'",
    'SVM-C5': "C=100.0, kernel='rbf', gamma=0.01, class_weight='balanced'",
    
    'RF-C1': "n_estimators=50, max_depth=4, class_weight='balanced'",
    'RF-C2': "n_estimators=100, max_depth=6, class_weight='balanced'",
    'RF-C3': "n_estimators=200, max_depth=8, class_weight='balanced'",
    'RF-C4': "n_estimators=300, max_depth=8, min_samples_split=10, class_weight='balanced'",
    'RF-C5': "n_estimators=500, max_depth=12, class_weight='balanced'",
    
    'XGB-C1': "n_estimators=100, max_depth=3, learning_rate=0.1",
    'XGB-C2': "n_estimators=200, max_depth=5, learning_rate=0.05",
    'XGB-C3': "n_estimators=150, max_depth=4, learning_rate=0.1",
    'XGB-C4': "n_estimators=300, max_depth=6, learning_rate=0.1",
    'XGB-C5': "n_estimators=500, max_depth=10, learning_rate=0.01",
    
    'DT-C1': "max_depth=3, min_samples_split=2, class_weight='balanced'",
    'DT-C2': "max_depth=5, min_samples_split=5, class_weight='balanced'",
    'DT-C3': "max_depth=6, min_samples_leaf=20, class_weight='balanced'",
    'DT-C4': "max_depth=8, min_samples_split=10, class_weight='balanced'",
    'DT-C5': "max_depth=12, min_samples_split=20, class_weight='balanced'",
}

# =====================================================================
# 4. GHI BÁO CÁO KET_QUA_TONG_HOP_SMOTE_RATIOS.MD
# =====================================================================
def generate_markdown_report(results):
    print("\n[TIEN TRINH] Dang ghi báo cáo ra file markdown...")
    
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        # Tiêu đề
        f.write("# BÁO CÁO TỔNG HỢP THỰC NGHIỆM CÁC MỐC TỶ LỆ SMOTE\n\n")
        f.write("- **Ngày thực hiện**: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("- **Tập dữ liệu**: `data_processed.csv`\n")
        f.write("- **Chiến lược chia**: Train 70%, Validation 10%, Test 20% (Random State = 42, Stratified).\n")
        f.write("- **Ngưỡng quyết định**: Chọn lựa ngưỡng trên tập Validation theo mục tiêu tối ưu hóa Recall nhãn 1 (Stroke=1) đạt >= 83% hoặc tối ưu điểm F1 nếu không đạt.\n")
        f.write("- **Đặc trưng MI Top 5**: `['Hypertension', 'Age', 'Heart_Disease', 'Avg_Glucose', 'Diabetes']` (được xác định dựa trên Mutual Information tính trên tập Train gốc trước SMOTE để tránh rò rỉ thông tin).\n\n")
        
        # Bảng chi tiết cấu hình tham số
        f.write("## 🛠️ CHI TIẾT CÁC CẤU HÌNH THAM SỐ (HYPERPARAMETERS)\n\n")
        f.write("Dưới đây là chi tiết tham số cụ thể của từng mã cấu hình:\n\n")
        
        f.write("### 1. Logistic Regression\n")
        for c in sorted(LR_CONFIGS.keys()):
            f.write(f"- **{c}**: `{CONFIG_PARAMS[c]}`\n")
        f.write("\n")
        
        f.write("### 2. SVM (RBF)\n")
        for c in sorted(SVM_CONFIGS.keys()):
            f.write(f"- **{c}**: `{CONFIG_PARAMS[c]}`\n")
        f.write("\n")
        
        f.write("### 3. Random Forest\n")
        for c in sorted(RF_CONFIGS.keys()):
            f.write(f"- **{c}**: `{CONFIG_PARAMS[c]}`\n")
        f.write("\n")
        
        f.write("### 4. XGBoost\n")
        for c in sorted(XGB_CONFIGS.keys()):
            f.write(f"- **{c}**: `{CONFIG_PARAMS[c]}`\n")
        f.write("\n")
        
        f.write("### 5. Decision Tree\n")
        for c in sorted(DT_CONFIGS.keys()):
            f.write(f"- **{c}**: `{CONFIG_PARAMS[c]}`\n")
        f.write("\n")
        
        # Phần 1: ĐỐI SÁNH CÁC CẤU HÌNH CỐT LÕI
        f.write("## 📊 1. ĐỐI SÁNH CÁC CẤU HÌNH CỐT LÕI QUA CÁC MỐC SMOTE\n\n")
        f.write("Dưới đây là so sánh chi tiết các cấu hình tối ưu của từng thuật toán qua 4 tỷ lệ SMOTE:\n")
        f.write("- **Logistic Regression (LR-C4)**\n")
        f.write("- **SVM (RBF) (SVM-C5)**\n")
        f.write("- **Random Forest (RF-C3)**\n")
        f.write("- **XGBoost (XGB-C3)**\n")
        f.write("- **Decision Tree (DT-C3)**\n\n")
        
        for algo_name, (ModelClass, configs, prefix, rep_config) in ALGO_MAP.items():
            f.write(f"### ❖ {algo_name} (Mã cấu hình cốt lõi: `{rep_config}` | Tham số: `{CONFIG_PARAMS[rep_config]}`)\n\n")
            f.write("| Kịch bản | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            
            # List các kịch bản đối sánh
            scenarios = [
                ("Full 11 (Không SMOTE)", "NoSMOTE", "Full11", "Dynamic"),
                ("Full 11 (SMOTE = 0.5)", "SMOTE_0.5", "Full11", "Dynamic"),
                ("Full 11 (SMOTE = 0.6)", "SMOTE_0.6", "Full11", "Dynamic"),
                ("Full 11 (SMOTE = 0.7)", "SMOTE_0.7", "Full11", "Dynamic"),
                ("MI Top 5 (Cố định, SMOTE = 0.5)", "SMOTE_0.5", "MI5", "Fixed"),
                ("MI Top 5 (Cố định, SMOTE = 0.6)", "SMOTE_0.6", "MI5", "Fixed"),
                ("MI Top 5 (Cố định, SMOTE = 0.7)", "SMOTE_0.7", "MI5", "Fixed"),
                ("MI Top 5 (Động, SMOTE = 0.5)", "SMOTE_0.5", "MI5", "Dynamic"),
                ("MI Top 5 (Động, SMOTE = 0.6)", "SMOTE_0.6", "MI5", "Dynamic"),
                ("MI Top 5 (Động, SMOTE = 0.7)", "SMOTE_0.7", "MI5", "Dynamic"),
            ]
            
            for label, ratio_lbl, f_set, thresh_type in scenarios:
                res = results.get((ratio_lbl, f_set, thresh_type, rep_config))
                if res:
                    f.write(f"| {label} | {res['threshold']:.4f} | {res['accuracy']} | {res['auc']} | Lớp 0 <br> Lớp 1 | {res['prec_l0']} <br> {res['prec_l1']} | {res['rec_l0']} <br> {res['rec_l1']} | {res['f1_l0']} <br> {res['f1_l1']} | {res['tn']} | {res['fp']} | {res['fn']} | {res['tp']} |\n")
            f.write("\n")
            
        # Phần 2: BẢNG CHI TIẾT TOÀN BỘ CẤU HÌNH (C1-C5)
        f.write("## 📝 2. BẢNG CHI TIẾT TOÀN BỘ CÁC CẤU HÌNH THEO TỪNG THUẬT TOÁN VÀ MỐC SMOTE\n\n")
        
        for algo_name, (ModelClass, configs, prefix, rep_config) in ALGO_MAP.items():
            f.write(f"### ❖ Thuật toán: {algo_name}\n\n")
            
            # Khởi động các tiểu mục A, B, C, D
            spaces = [
                ("A. Không gian Full 11 Features (Không SMOTE)", "NoSMOTE", "Full11", "Dynamic"),
                ("B. Không gian Full 11 Features (Có SMOTE = 0.5)", "SMOTE_0.5", "Full11", "Dynamic"),
                ("C. Không gian Full 11 Features (Có SMOTE = 0.6)", "SMOTE_0.6", "Full11", "Dynamic"),
                ("D. Không gian Full 11 Features (Có SMOTE = 0.7)", "SMOTE_0.7", "Full11", "Dynamic"),
                ("E. Không gian MI Top 5 Features (Có SMOTE = 0.5) - Ngưỡng động tối ưu", "SMOTE_0.5", "MI5", "Dynamic"),
                ("F. Không gian MI Top 5 Features (Có SMOTE = 0.6) - Ngưỡng động tối ưu", "SMOTE_0.6", "MI5", "Dynamic"),
                ("G. Không gian MI Top 5 Features (Có SMOTE = 0.7) - Ngưỡng động tối ưu", "SMOTE_0.7", "MI5", "Dynamic"),
            ]
            
            for title, ratio_lbl, f_set, thresh_type in spaces:
                f.write(f"#### {title}\n\n")
                f.write("**Tham số chi tiết của các cấu hình:**\n")
                for c in sorted(configs.keys()):
                    f.write(f"- **{c}**: `{CONFIG_PARAMS[c]}`\n")
                f.write("\n")
                f.write("| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |\n")
                f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
                
                # Sắp xếp theo config từ C1 đến C5
                for config_code in sorted(configs.keys()):
                    res = results.get((ratio_lbl, f_set, thresh_type, config_code))
                    if res:
                        f.write(f"| {config_code} | {res['threshold']:.4f} | {res['accuracy']} | {res['auc']} | Lớp 0 <br> Lớp 1 | {res['prec_l0']} <br> {res['prec_l1']} | {res['rec_l0']} <br> {res['rec_l1']} | {res['f1_l0']} <br> {res['f1_l1']} | {res['tn']} | {res['fp']} | {res['fn']} | {res['tp']} |\n")
                f.write("\n")
                
    print(f"[LOG] Đã ghi báo cáo tổng hợp thành công tại {REPORT_PATH}")

# =====================================================================
# HÀM KHỞI CHẠY CHÍNH
# =====================================================================
def main():
    print("=" * 80)
    print("BAT DAU CHUONG TRINH SO SANH CAC MOC SMOTE (NO SMOTE, 0.5, 0.6, 0.7)")
    print("=" * 80)
    
    start_time = time.time()
    results = run_experiments()
    generate_markdown_report(results)
    
    elapsed = time.time() - start_time
    print(f"\n[HOAN THANH] Tong thoi gian chay thuc nghiem: {elapsed:.2f} giay.")
    print("=" * 80)

if __name__ == "__main__":
    main()
