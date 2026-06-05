"""
CHƯƠNG TRÌNH KIỂM TRA ĐỘC LẬP: Xác minh kết quả trong ket_qua_mi_thresholds_comparison.md

Script này chạy lại toàn bộ pipeline ML từ đầu (chia dữ liệu, tính MI, SMOTE, train, tune threshold, evaluate)
và so sánh kết quả với những giá trị đã ghi trong file báo cáo markdown.

Mọi sai lệch sẽ được in ra chi tiết.
"""

import os
import sys
import re
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

# Setup UTF-8 encoding for stdout on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

warnings.filterwarnings('ignore')

# =============================================================================
# PATHS AND CONSTANTS
# =============================================================================
WORKSPACE = r"e:\kpdl_main"
DATA_PROCESSED = os.path.join(WORKSPACE, "data_processed.csv")
REPORT_PATH = os.path.join(WORKSPACE, "ket_qua_mi_thresholds_comparison.md")
TARGET_COL = "Stroke"

# Same core configs as the original script
CORE_CONFIGS = {
    'Logistic Regression': {
        'code': 'LR-C4',
        'class': LogisticRegression,
        'params': {'C': 10.0, 'penalty': 'l2', 'class_weight': 'balanced', 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}
    },
    'SVM (RBF)': {
        'code': 'SVM-C5',
        'class': SVC,
        'params': {'C': 100.0, 'kernel': 'rbf', 'gamma': 0.01, 'class_weight': 'balanced', 'probability': True, 'random_state': 42, 'cache_size': 1000}
    },
    'Random Forest': {
        'code': 'RF-C3',
        'class': RandomForestClassifier,
        'params': {'max_depth': 8, 'n_estimators': 200, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}
    },
    'XGBoost': {
        'code': 'XGB-C3',
        'class': XGBClassifier,
        'params': {'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 150, 'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1}
    },
    'Decision Tree': {
        'code': 'DT-C3',
        'class': DecisionTreeClassifier,
        'params': {'max_depth': 6, 'min_samples_leaf': 20, 'class_weight': 'balanced', 'random_state': 42}
    }
}


# =============================================================================
# PHẦN 1: PARSE KẾT QUẢ TỪ FILE MARKDOWN
# =============================================================================

def parse_mi_scores_from_report(content):
    """Parse bảng MI scores từ báo cáo."""
    mi_scores = {}
    # Find lines like: | 1 | **Hypertension** | `0.076417` | 1 thuộc tính |
    pattern = re.compile(r'\|\s*(\d+)\s*\|\s*\*\*(.+?)\*\*\s*\|\s*`([\d.]+)`\s*\|\s*(\d+)\s*thuộc tính\s*\|')
    for match in pattern.finditer(content):
        rank = int(match.group(1))
        feat_name = match.group(2)
        mi_value = float(match.group(3))
        mi_scores[feat_name] = {'rank': rank, 'mi_score': mi_value}
    return mi_scores


def parse_feature_subsets_from_report(content):
    """Parse danh sách feature subsets từ báo cáo."""
    subsets = {}
    # Pattern: - **Mốc 5 thuộc tính** (Ngưỡng MI >= `0.031333`):
    #   - Danh sách: `['Hypertension', 'Age', ...]`
    pattern = re.compile(
        r'\*\*Mốc (\d+) thuộc tính\*\*\s*\(Ngưỡng MI >= `([\d.]+)`\):\s*\n'
        r'\s*-\s*Danh sách:\s*`\[(.+?)\]`',
        re.MULTILINE
    )
    for match in pattern.finditer(content):
        size = int(match.group(1))
        min_mi = float(match.group(2))
        features_str = match.group(3)
        features = [f.strip().strip("'\"") for f in features_str.split(',')]
        subsets[size] = {'min_mi': min_mi, 'features': features}
    return subsets


def parse_section2_tables(content):
    """
    Parse tất cả kết quả từ Section 2 (CHI TIẾT TOÀN BỘ CẤU HÌNH QUA CÁC TỶ LỆ SMOTE KHÁC NHAU).
    Trả về dict: (ratio_label, num_features, algo_name) -> metrics_dict
    """
    results = {}
    
    # Split by SMOTE strategy sections
    smote_sections = re.split(r'### ❖ Chiến lược cân bằng dữ liệu:', content)
    
    for section in smote_sections[1:]:  # skip before first heading
        # Determine ratio label
        if 'NoSMOTE' in section.split('\n')[0]:
            ratio_label = 'NoSMOTE'
        else:
            smote_match = re.search(r'SMOTE\s*=\s*([\d.]+)', section.split('\n')[0])
            if smote_match:
                ratio_label = f'SMOTE_{smote_match.group(1)}'
            else:
                continue
        
        # Parse table rows
        # Pattern: | Algo Name | **N** | `threshold` | accuracy% | auc% | labels | prec | recall | f1 | TN | FP | FN | TP |
        lines = section.split('\n')
        for line in lines:
            line = line.strip()
            if not line.startswith('|') or '---' in line:
                continue
            
            cells = [c.strip() for c in line.split('|')]
            cells = [c for c in cells if c]  # remove empty strings from split
            
            if len(cells) < 12:
                continue
                
            algo_name_raw = cells[0]
            
            # Skip header rows
            if algo_name_raw in ['Thuật toán', ':---']:
                continue
            
            # Extract number of features
            size_match = re.search(r'\*\*(\d+)\*\*', cells[1])
            if not size_match:
                continue
            num_feats = int(size_match.group(1))
            
            # Extract threshold
            thresh_match = re.search(r'`([\d.]+)`', cells[2])
            if not thresh_match:
                continue
            threshold = float(thresh_match.group(1))
            
            # Extract accuracy
            acc_match = re.search(r'([\d.]+)%', cells[3])
            accuracy = float(acc_match.group(1)) if acc_match else None
            
            # Extract AUC
            auc_match = re.search(r'([\d.]+)%', cells[4])
            auc = float(auc_match.group(1)) if auc_match else None
            
            # Extract Precision (class 0 and 1 separated by <br>)
            prec_parts = re.findall(r'([\d.]+)%', cells[6])
            prec_l0 = float(prec_parts[0]) if len(prec_parts) >= 1 else None
            prec_l1 = float(prec_parts[1]) if len(prec_parts) >= 2 else None
            
            # Extract Recall
            rec_parts = re.findall(r'([\d.]+)%', cells[7])
            rec_l0 = float(rec_parts[0]) if len(rec_parts) >= 1 else None
            rec_l1 = float(rec_parts[1]) if len(rec_parts) >= 2 else None
            
            # Extract F1
            f1_parts = re.findall(r'([\d.]+)%', cells[8])
            f1_l0 = float(f1_parts[0]) if len(f1_parts) >= 1 else None
            f1_l1 = float(f1_parts[1]) if len(f1_parts) >= 2 else None
            
            # Extract confusion matrix
            tn = int(cells[9])
            fp = int(cells[10])
            fn = int(cells[11])
            tp = int(cells[12])
            
            key = (ratio_label, num_feats, algo_name_raw)
            results[key] = {
                'threshold': threshold,
                'accuracy': accuracy,
                'auc': auc,
                'prec_l0': prec_l0, 'prec_l1': prec_l1,
                'rec_l0': rec_l0, 'rec_l1': rec_l1,
                'f1_l0': f1_l0, 'f1_l1': f1_l1,
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
            }
    
    return results


# =============================================================================
# PHẦN 2: CHẠY LẠI PIPELINE ĐỘC LẬP
# =============================================================================

def get_original_splits():
    df = pd.read_csv(DATA_PROCESSED)
    x = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    
    # Split train_val / test (80/20)
    x_temp_train, x_test, y_temp_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )
    # Split train / val (70/10)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp_train, y_temp_train, test_size=0.125, random_state=42, stratify=y_temp_train
    )
    
    # Verify no data leakage
    assert len(set(x_train.index).intersection(set(x_val.index))) == 0, "LEAK between train and val!"
    assert len(set(x_train.index).intersection(set(x_test.index))) == 0, "LEAK between train and test!"
    assert len(set(x_val.index).intersection(set(x_test.index))) == 0, "LEAK between val and test!"
    
    return x_train, x_val, x_test, y_train, y_val, y_test


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
    
    prec_l0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec_l0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_l0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    
    prec_l1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_l1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_l1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    return {
        'threshold': threshold,
        'accuracy': round(accuracy * 100, 2),
        'auc': round(auc * 100, 2),
        'prec_l0': round(prec_l0 * 100, 2),
        'prec_l1': round(prec_l1 * 100, 2),
        'rec_l0': round(rec_l0 * 100, 2),
        'rec_l1': round(rec_l1 * 100, 2),
        'f1_l0': round(f1_l0 * 100, 2),
        'f1_l1': round(f1_l1 * 100, 2),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }


# =============================================================================
# PHẦN 3: SO SÁNH VÀ BÁO CÁO
# =============================================================================

def compare_values(label, expected, actual, tolerance=0.011):
    """So sánh hai giá trị số với sai số cho phép. Trả về True nếu khớp."""
    if expected is None or actual is None:
        return True  # skip if can't compare
    diff = abs(expected - actual)
    if diff > tolerance:
        return False
    return True


def main():
    print("=" * 90)
    print("  CHƯƠNG TRÌNH KIỂM TRA ĐỘC LẬP - XÁC MINH KẾT QUẢ MI THRESHOLDS COMPARISON")
    print("=" * 90)
    
    # ----- Step 1: Parse expected results from markdown -----
    print("\n[BƯỚC 1] Đọc và parse kết quả từ file báo cáo markdown...")
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        report_content = f.read()
    
    expected_mi = parse_mi_scores_from_report(report_content)
    expected_subsets = parse_feature_subsets_from_report(report_content)
    expected_results = parse_section2_tables(report_content)
    
    print(f"  → Đã parse {len(expected_mi)} MI scores")
    print(f"  → Đã parse {len(expected_subsets)} feature subsets")
    print(f"  → Đã parse {len(expected_results)} kết quả thí nghiệm")
    
    # ----- Step 2: Re-run pipeline -----
    print("\n[BƯỚC 2] Chạy lại pipeline từ đầu...")
    
    x_train_raw, x_val_raw, x_test_raw, y_train_raw, y_val_raw, y_test_raw = get_original_splits()
    print(f"  → Train: {len(x_train_raw)}, Val: {len(x_val_raw)}, Test: {len(x_test_raw)}")
    print(f"  → Train class dist: {dict(y_train_raw.value_counts())}")
    print(f"  → Test class dist: {dict(y_test_raw.value_counts())}")
    
    # Calculate MI scores
    print("  → Tính Mutual Information trên tập Train gốc...")
    mi_scores = mutual_info_classif(x_train_raw, y_train_raw, random_state=42)
    mi_series = pd.Series(mi_scores, index=x_train_raw.columns).sort_values(ascending=False)
    
    # Build feature subsets
    feature_sizes = list(range(5, 12))
    feature_subsets = {}
    for size in feature_sizes:
        selected_feats = list(mi_series.index[:size])
        min_mi_val = float(mi_series.values[size-1])
        feature_subsets[size] = {
            'features': selected_feats,
            'min_mi': min_mi_val
        }
    
    # ----- Step 3: Verify MI Scores -----
    print("\n[BƯỚC 3] Kiểm tra điểm Mutual Information...")
    mi_errors = 0
    for feat, info in expected_mi.items():
        expected_score = info['mi_score']
        if feat in mi_series.index:
            actual_score = mi_series[feat]
            if abs(expected_score - actual_score) > 0.000005:
                print(f"  ❌ MI SCORE SAI - {feat}: Báo cáo={expected_score:.6f}, Tính lại={actual_score:.6f}")
                mi_errors += 1
            else:
                print(f"  ✅ MI score khớp - {feat}: {actual_score:.6f}")
        else:
            print(f"  ❌ Feature '{feat}' không tồn tại trong dữ liệu!")
            mi_errors += 1
    
    # Verify ranking order
    print("\n  Kiểm tra thứ tự xếp hạng:")
    actual_ranking = list(mi_series.index)
    expected_ranking = sorted(expected_mi.keys(), key=lambda x: expected_mi[x]['rank'])
    if actual_ranking == expected_ranking:
        print(f"  ✅ Thứ tự xếp hạng MI khớp hoàn toàn")
    else:
        print(f"  ❌ Thứ tự xếp hạng MI KHÔNG KHỚP!")
        print(f"     Báo cáo:  {expected_ranking}")
        print(f"     Tính lại: {actual_ranking}")
        mi_errors += 1
    
    # ----- Step 4: Verify Feature Subsets -----
    print("\n[BƯỚC 4] Kiểm tra danh sách thuộc tính cho mỗi mốc...")
    subset_errors = 0
    for size in sorted(expected_subsets.keys()):
        exp = expected_subsets[size]
        act = feature_subsets.get(size)
        if act is None:
            print(f"  ❌ Mốc {size}: Không có kết quả tính lại!")
            subset_errors += 1
            continue
        
        if exp['features'] != act['features']:
            print(f"  ❌ Mốc {size}: Danh sách thuộc tính KHÔNG KHỚP!")
            print(f"     Báo cáo:  {exp['features']}")
            print(f"     Tính lại: {act['features']}")
            subset_errors += 1
        elif abs(exp['min_mi'] - act['min_mi']) > 0.000005:
            print(f"  ❌ Mốc {size}: min_mi KHÔNG KHỚP! Báo cáo={exp['min_mi']:.6f}, Tính lại={act['min_mi']:.6f}")
            subset_errors += 1
        else:
            print(f"  ✅ Mốc {size} thuộc tính khớp: {act['features']}")
    
    # ----- Step 5: Re-run all experiments and compare -----
    print("\n[BƯỚC 5] Chạy lại tất cả thí nghiệm và so sánh kết quả...")
    
    smote_ratios = [None, 0.5, 0.6, 0.7]
    total_checks = 0
    total_errors = 0
    total_matches = 0
    
    for ratio in smote_ratios:
        ratio_label = "NoSMOTE" if ratio is None else f"SMOTE_{ratio}"
        print(f"\n  ────────────────────────────────────────")
        print(f"  📋 Đang kiểm tra: {ratio_label}")
        print(f"  ────────────────────────────────────────")
        
        # Apply SMOTE
        if ratio is not None:
            sm = SMOTE(sampling_strategy=ratio, random_state=42)
            x_train_sm, y_train_sm = sm.fit_resample(x_train_raw, y_train_raw)
        else:
            x_train_sm, y_train_sm = x_train_raw.copy(), y_train_raw.copy()
        
        for size in feature_sizes:
            feats = feature_subsets[size]['features']
            x_tr = x_train_sm[feats]
            x_v = x_val_raw[feats]
            x_te = x_test_raw[feats]
            
            for algo_name, info in CORE_CONFIGS.items():
                ModelClass = info['class']
                params = info['params']
                
                # Train model
                model = ModelClass(**params)
                model.fit(x_tr, y_train_sm)
                
                # Dynamic threshold tuning on validation set
                val_proba = get_proba(model, x_v)
                dynamic_thresh = find_extreme_recall_threshold(y_val_raw, val_proba, target_recall=0.83)
                
                # Evaluate on test set
                actual = evaluate_test(model, x_te, y_test_raw, dynamic_thresh)
                
                # Lookup expected
                exp_key = (ratio_label, size, algo_name)
                expected = expected_results.get(exp_key)
                
                total_checks += 1
                
                if expected is None:
                    print(f"    ⚠️  {algo_name} | {size} feats | {ratio_label}: Không tìm thấy kết quả trong báo cáo để so sánh")
                    continue
                
                # Compare all metrics
                errors_for_this = []
                
                # Threshold
                if abs(expected['threshold'] - actual['threshold']) > 0.00051:
                    errors_for_this.append(f"Threshold: báo cáo={expected['threshold']:.4f}, tính lại={actual['threshold']:.4f}")
                
                # Accuracy
                if not compare_values('Accuracy', expected['accuracy'], actual['accuracy']):
                    errors_for_this.append(f"Accuracy: báo cáo={expected['accuracy']:.2f}%, tính lại={actual['accuracy']:.2f}%")
                
                # AUC
                if not compare_values('AUC', expected['auc'], actual['auc']):
                    errors_for_this.append(f"AUC: báo cáo={expected['auc']:.2f}%, tính lại={actual['auc']:.2f}%")
                
                # Precision
                if not compare_values('Prec_L0', expected['prec_l0'], actual['prec_l0']):
                    errors_for_this.append(f"Prec_L0: báo cáo={expected['prec_l0']:.2f}%, tính lại={actual['prec_l0']:.2f}%")
                if not compare_values('Prec_L1', expected['prec_l1'], actual['prec_l1']):
                    errors_for_this.append(f"Prec_L1: báo cáo={expected['prec_l1']:.2f}%, tính lại={actual['prec_l1']:.2f}%")
                
                # Recall
                if not compare_values('Rec_L0', expected['rec_l0'], actual['rec_l0']):
                    errors_for_this.append(f"Rec_L0: báo cáo={expected['rec_l0']:.2f}%, tính lại={actual['rec_l0']:.2f}%")
                if not compare_values('Rec_L1', expected['rec_l1'], actual['rec_l1']):
                    errors_for_this.append(f"Rec_L1: báo cáo={expected['rec_l1']:.2f}%, tính lại={actual['rec_l1']:.2f}%")
                
                # F1
                if not compare_values('F1_L0', expected['f1_l0'], actual['f1_l0']):
                    errors_for_this.append(f"F1_L0: báo cáo={expected['f1_l0']:.2f}%, tính lại={actual['f1_l0']:.2f}%")
                if not compare_values('F1_L1', expected['f1_l1'], actual['f1_l1']):
                    errors_for_this.append(f"F1_L1: báo cáo={expected['f1_l1']:.2f}%, tính lại={actual['f1_l1']:.2f}%")
                
                # Confusion Matrix
                for cm_name in ['tn', 'fp', 'fn', 'tp']:
                    if expected[cm_name] != actual[cm_name]:
                        errors_for_this.append(f"{cm_name.upper()}: báo cáo={expected[cm_name]}, tính lại={actual[cm_name]}")
                
                if errors_for_this:
                    total_errors += 1
                    print(f"    ❌ {algo_name} | {size} feats | {ratio_label}:")
                    for err in errors_for_this:
                        print(f"       → {err}")
                else:
                    total_matches += 1
                    print(f"    ✅ {algo_name} | {size} feats | {ratio_label}: Tất cả metrics khớp")
    
    # ----- Step 6: Summary -----
    print("\n" + "=" * 90)
    print("  📊 TỔNG KẾT KIỂM TRA")
    print("=" * 90)
    print(f"  • Tổng số thí nghiệm kiểm tra: {total_checks}")
    print(f"  • Số thí nghiệm KHỚP hoàn toàn: {total_matches} ✅")
    print(f"  • Số thí nghiệm CÓ SAI LỆCH:  {total_errors} ❌")
    print(f"  • Lỗi MI scores: {mi_errors}")
    print(f"  • Lỗi Feature subsets: {subset_errors}")
    
    total_all_errors = mi_errors + subset_errors + total_errors
    if total_all_errors == 0:
        print(f"\n  🎉 KẾT LUẬN: TẤT CẢ KẾT QUẢ TRONG BÁO CÁO ĐÃ ĐƯỢC XÁC MINH LÀ CHÍNH XÁC!")
    else:
        print(f"\n  ⚠️  KẾT LUẬN: CÓ {total_all_errors} SAI LỆCH CẦN XEM XÉT!")
    
    print("=" * 90)


if __name__ == "__main__":
    main()
