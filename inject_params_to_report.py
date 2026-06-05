import os
import re

REPORT_PATH = r"e:\kpdl_main\ket_qua_tong_hop_smote_ratios.md"

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

with open(REPORT_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Inject CHI TIẾT CÁC CẤU HÌNH THAM SỐ at the top
param_section = """## 🛠️ CHI TIẾT CÁC CẤU HÌNH THAM SỐ (HYPERPARAMETERS)

Dưới đây là chi tiết tham số cụ thể của từng mã cấu hình:

### 1. Logistic Regression
- **LR-C1**: `C=0.01, penalty='l2', class_weight='balanced'`
- **LR-C2**: `C=0.1, penalty='l2', class_weight='balanced'`
- **LR-C3**: `C=1.0, penalty='l2', class_weight='balanced'`
- **LR-C4**: `C=10.0, penalty='l2', class_weight='balanced'`
- **LR-C5**: `C=100.0, penalty='l2', class_weight='balanced'`

### 2. SVM (RBF)
- **SVM-C1**: `C=0.1, kernel='rbf', gamma='scale', class_weight='balanced'`
- **SVM-C2**: `C=1.0, kernel='rbf', gamma='scale', class_weight='balanced'`
- **SVM-C3**: `C=10.0, kernel='rbf', gamma=0.01, class_weight='balanced'`
- **SVM-C4**: `C=10.0, kernel='rbf', gamma=0.1, class_weight='balanced'`
- **SVM-C5**: `C=100.0, kernel='rbf', gamma=0.01, class_weight='balanced'`

### 3. Random Forest
- **RF-C1**: `n_estimators=50, max_depth=4, class_weight='balanced'`
- **RF-C2**: `n_estimators=100, max_depth=6, class_weight='balanced'`
- **RF-C3**: `n_estimators=200, max_depth=8, class_weight='balanced'`
- **RF-C4**: `n_estimators=300, max_depth=8, min_samples_split=10, class_weight='balanced'`
- **RF-C5**: `n_estimators=500, max_depth=12, class_weight='balanced'`

### 4. XGBoost
- **XGB-C1**: `n_estimators=100, max_depth=3, learning_rate=0.1`
- **XGB-C2**: `n_estimators=200, max_depth=5, learning_rate=0.05`
- **XGB-C3**: `n_estimators=150, max_depth=4, learning_rate=0.1`
- **XGB-C4**: `n_estimators=300, max_depth=6, learning_rate=0.1`
- **XGB-C5**: `n_estimators=500, max_depth=10, learning_rate=0.01`

### 5. Decision Tree
- **DT-C1**: `max_depth=3, min_samples_split=2, class_weight='balanced'`
- **DT-C2**: `max_depth=5, min_samples_split=5, class_weight='balanced'`
- **DT-C3**: `max_depth=6, min_samples_leaf=20, class_weight='balanced'`
- **DT-C4**: `max_depth=8, min_samples_split=10, class_weight='balanced'`
- **DT-C5**: `max_depth=12, min_samples_split=20, class_weight='balanced'`

"""

# Insert param_section right before "## 📊 1. ĐỐI SÁNH CÁC CẤU HÌNH CỐT LÕI QUA CÁC MỐC SMOTE"
content = content.replace("## 📊 1. ĐỐI SÁNH CÁC CẤU HÌNH CỐT LÕI QUA CÁC MỐC SMOTE", param_section + "## 📊 1. ĐỐI SÁNH CÁC CẤU HÌNH CỐT LÕI QUA CÁC MỐC SMOTE")

# 2. Modify Section 1 header lines
content = content.replace("### ❖ Logistic Regression (Mã cấu hình cốt lõi: `LR-C4`)", "### ❖ Logistic Regression (Mã cấu hình cốt lõi: `LR-C4` | Tham số: `C=10.0, penalty='l2', class_weight='balanced'`)")
content = content.replace("### ❖ SVM (RBF) (Mã cấu hình cốt lõi: `SVM-C5`)", "### ❖ SVM (RBF) (Mã cấu hình cốt lõi: `SVM-C5` | Tham số: `C=100.0, kernel='rbf', gamma=0.01, class_weight='balanced'`)")
content = content.replace("### ❖ Random Forest (Mã cấu hình cốt lõi: `RF-C3`)", "### ❖ Random Forest (Mã cấu hình cốt lõi: `RF-C3` | Tham số: `n_estimators=200, max_depth=8, class_weight='balanced'`)")
content = content.replace("### ❖ XGBoost (Mã cấu hình cốt lõi: `XGB-C3`)", "### ❖ XGBoost (Mã cấu hình cốt lõi: `XGB-C3` | Tham số: `n_estimators=150, max_depth=4, learning_rate=0.1`)")
content = content.replace("### ❖ Decision Tree (Mã cấu hình cốt lõi: `DT-C3`)", "### ❖ Decision Tree (Mã cấu hình cốt lõi: `DT-C3` | Tham số: `max_depth=6, min_samples_leaf=20, class_weight='balanced'`)")

# 3. Modify Section 2 subheadings by appending the parameters list below them
subheadings = {
    "Logistic Regression": ["LR-C1", "LR-C2", "LR-C3", "LR-C4", "LR-C5"],
    "SVM (RBF)": ["SVM-C1", "SVM-C2", "SVM-C3", "SVM-C4", "SVM-C5"],
    "Random Forest": ["RF-C1", "RF-C2", "RF-C3", "RF-C4", "RF-C5"],
    "XGBoost": ["XGB-C1", "XGB-C2", "XGB-C3", "XGB-C4", "XGB-C5"],
    "Decision Tree": ["DT-C1", "DT-C2", "DT-C3", "DT-C4", "DT-C5"]
}

lines = content.split('\n')
new_lines = []
current_algo = None

for line in lines:
    new_lines.append(line)
    # Check if this line marks the start of an algorithm's detailed section
    if line.startswith("### ❖ Thuật toán: "):
        current_algo = line.replace("### ❖ Thuật toán: ", "").strip()
    elif line.startswith("#### ") and current_algo in subheadings:
        # We are at a subheading like: #### A. Không gian Full 11 Features (Không SMOTE)
        # We insert the parameter list right after it
        param_list = ["", "**Tham số chi tiết của các cấu hình:**"]
        for c in subheadings[current_algo]:
            param_list.append(f"- **{c}**: `{CONFIG_PARAMS[c]}`")
        new_lines.extend(param_list)

content = '\n'.join(new_lines)

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write(content)

print("SUCCESS: Injected parameters successfully!")
