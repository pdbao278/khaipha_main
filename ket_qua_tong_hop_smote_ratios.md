# BÁO CÁO TỔNG HỢP THỰC NGHIỆM CÁC MỐC TỶ LỆ SMOTE

- **Ngày thực hiện**: 2026-06-05 08:35:35
- **Tập dữ liệu**: `data_processed.csv`
- **Chiến lược chia**: Train 70%, Validation 10%, Test 20% (Random State = 42, Stratified).
- **Ngưỡng quyết định**: Chọn lựa ngưỡng trên tập Validation theo mục tiêu tối ưu hóa Recall nhãn 1 (Stroke=1) đạt >= 83% hoặc tối ưu điểm F1 nếu không đạt.
- **Đặc trưng MI Top 5**: `['Hypertension', 'Age', 'Heart_Disease', 'Avg_Glucose', 'Diabetes']` (được xác định dựa trên Mutual Information tính trên tập Train gốc trước SMOTE để tránh rò rỉ thông tin).

## 🛠️ CHI TIẾT CÁC CẤU HÌNH THAM SỐ (HYPERPARAMETERS)

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

## 📊 1. ĐỐI SÁNH CÁC CẤU HÌNH CỐT LÕI QUA CÁC MỐC SMOTE

Dưới đây là so sánh chi tiết các cấu hình tối ưu của từng thuật toán qua 4 tỷ lệ SMOTE:
- **Logistic Regression (LR-C4)**
- **SVM (RBF) (SVM-C5)**
- **Random Forest (RF-C3)**
- **XGBoost (XGB-C3)**
- **Decision Tree (DT-C3)**

### ❖ Logistic Regression (Mã cấu hình cốt lõi: `LR-C4` | Tham số: `C=10.0, penalty='l2', class_weight='balanced'`)

| Kịch bản | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Full 11 (Không SMOTE) | 0.4440 | 73.95% | 85.11% | Lớp 0 <br> Lớp 1 | 89.60% <br> 54.24% | 71.15% <br> 80.54% | 79.32% <br> 64.82% | 999 | 405 | 116 | 480 |
| Full 11 (SMOTE = 0.5) | 0.4455 | 73.90% | 85.07% | Lớp 0 <br> Lớp 1 | 89.80% <br> 54.15% | 70.87% <br> 81.04% | 79.22% <br> 64.92% | 995 | 409 | 113 | 483 |
| Full 11 (SMOTE = 0.6) | 0.4490 | 74.05% | 85.07% | Lớp 0 <br> Lớp 1 | 89.83% <br> 54.33% | 71.08% <br> 81.04% | 79.36% <br> 65.05% | 998 | 406 | 113 | 483 |
| Full 11 (SMOTE = 0.7) | 0.4670 | 74.50% | 85.07% | Lớp 0 <br> Lớp 1 | 89.21% <br> 55.00% | 72.44% <br> 79.36% | 79.95% <br> 64.97% | 1017 | 387 | 123 | 473 |
| MI Top 5 (Cố định, SMOTE = 0.5) | 0.4455 | 73.80% | 84.63% | Lớp 0 <br> Lớp 1 | 90.44% <br> 53.95% | 70.09% <br> 82.55% | 78.97% <br> 65.25% | 984 | 420 | 104 | 492 |
| MI Top 5 (Cố định, SMOTE = 0.6) | 0.4490 | 73.85% | 84.64% | Lớp 0 <br> Lớp 1 | 90.45% <br> 54.01% | 70.16% <br> 82.55% | 79.02% <br> 65.30% | 985 | 419 | 104 | 492 |
| MI Top 5 (Cố định, SMOTE = 0.7) | 0.4670 | 74.90% | 84.62% | Lớp 0 <br> Lớp 1 | 90.20% <br> 55.35% | 72.08% <br> 81.54% | 80.13% <br> 65.94% | 1012 | 392 | 110 | 486 |
| MI Top 5 (Động, SMOTE = 0.5) | 0.4480 | 74.00% | 84.63% | Lớp 0 <br> Lớp 1 | 90.48% <br> 54.19% | 70.37% <br> 82.55% | 79.17% <br> 65.43% | 988 | 416 | 104 | 492 |
| MI Top 5 (Động, SMOTE = 0.6) | 0.4430 | 73.45% | 84.64% | Lớp 0 <br> Lớp 1 | 90.45% <br> 53.53% | 69.52% <br> 82.72% | 78.61% <br> 65.00% | 976 | 428 | 103 | 493 |
| MI Top 5 (Động, SMOTE = 0.7) | 0.4550 | 73.90% | 84.62% | Lớp 0 <br> Lớp 1 | 90.31% <br> 54.08% | 70.37% <br> 82.21% | 79.10% <br> 65.25% | 988 | 416 | 106 | 490 |

### ❖ SVM (RBF) (Mã cấu hình cốt lõi: `SVM-C5` | Tham số: `C=100.0, kernel='rbf', gamma=0.01, class_weight='balanced'`)

| Kịch bản | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Full 11 (Không SMOTE) | 0.2040 | 73.85% | 84.77% | Lớp 0 <br> Lớp 1 | 89.65% <br> 54.11% | 70.94% <br> 80.70% | 79.20% <br> 64.78% | 996 | 408 | 115 | 481 |
| Full 11 (SMOTE = 0.5) | 0.1765 | 70.05% | 84.64% | Lớp 0 <br> Lớp 1 | 91.20% <br> 49.85% | 63.46% <br> 85.57% | 74.84% <br> 63.00% | 891 | 513 | 86 | 510 |
| Full 11 (SMOTE = 0.6) | 0.2530 | 73.30% | 84.61% | Lớp 0 <br> Lớp 1 | 89.62% <br> 53.44% | 70.09% <br> 80.87% | 78.66% <br> 64.35% | 984 | 420 | 114 | 482 |
| Full 11 (SMOTE = 0.7) | 0.2890 | 72.75% | 84.59% | Lớp 0 <br> Lớp 1 | 89.37% <br> 52.81% | 69.44% <br> 80.54% | 78.16% <br> 63.79% | 975 | 429 | 116 | 480 |
| MI Top 5 (Cố định, SMOTE = 0.5) | 0.1765 | 70.05% | 83.98% | Lớp 0 <br> Lớp 1 | 91.45% <br> 49.85% | 63.25% <br> 86.07% | 74.78% <br> 63.14% | 888 | 516 | 83 | 513 |
| MI Top 5 (Cố định, SMOTE = 0.6) | 0.2530 | 75.95% | 83.86% | Lớp 0 <br> Lớp 1 | 88.49% <br> 57.18% | 75.57% <br> 76.85% | 81.52% <br> 65.57% | 1061 | 343 | 138 | 458 |
| MI Top 5 (Cố định, SMOTE = 0.7) | 0.2890 | 74.75% | 83.87% | Lớp 0 <br> Lớp 1 | 89.19% <br> 55.33% | 72.86% <br> 79.19% | 80.20% <br> 65.15% | 1023 | 381 | 124 | 472 |
| MI Top 5 (Động, SMOTE = 0.5) | 0.2155 | 74.55% | 83.98% | Lớp 0 <br> Lớp 1 | 89.71% <br> 54.98% | 72.01% <br> 80.54% | 79.89% <br> 65.35% | 1011 | 393 | 116 | 480 |
| MI Top 5 (Động, SMOTE = 0.6) | 0.2245 | 73.70% | 83.86% | Lớp 0 <br> Lớp 1 | 89.27% <br> 53.97% | 71.08% <br> 79.87% | 79.14% <br> 64.41% | 998 | 406 | 120 | 476 |
| MI Top 5 (Động, SMOTE = 0.7) | 0.2565 | 72.60% | 83.87% | Lớp 0 <br> Lớp 1 | 89.85% <br> 52.59% | 68.73% <br> 81.71% | 77.89% <br> 63.99% | 965 | 439 | 109 | 487 |

### ❖ Random Forest (Mã cấu hình cốt lõi: `RF-C3` | Tham số: `n_estimators=200, max_depth=8, class_weight='balanced'`)

| Kịch bản | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Full 11 (Không SMOTE) | 0.4145 | 73.50% | 84.65% | Lớp 0 <br> Lớp 1 | 89.58% <br> 53.68% | 70.44% <br> 80.70% | 78.87% <br> 64.48% | 989 | 415 | 115 | 481 |
| Full 11 (SMOTE = 0.5) | 0.3875 | 71.05% | 84.50% | Lớp 0 <br> Lớp 1 | 90.64% <br> 50.86% | 65.53% <br> 84.06% | 76.06% <br> 63.38% | 920 | 484 | 95 | 501 |
| Full 11 (SMOTE = 0.6) | 0.4215 | 73.40% | 84.58% | Lớp 0 <br> Lớp 1 | 89.14% <br> 53.61% | 70.73% <br> 79.70% | 78.87% <br> 64.10% | 993 | 411 | 121 | 475 |
| Full 11 (SMOTE = 0.7) | 0.4110 | 72.35% | 84.53% | Lớp 0 <br> Lớp 1 | 89.58% <br> 52.32% | 68.59% <br> 81.21% | 77.69% <br> 63.64% | 963 | 441 | 112 | 484 |
| MI Top 5 (Cố định, SMOTE = 0.5) | 0.3875 | 72.65% | 82.77% | Lớp 0 <br> Lớp 1 | 88.71% <br> 52.74% | 69.94% <br> 79.03% | 78.22% <br> 63.26% | 982 | 422 | 125 | 471 |
| MI Top 5 (Cố định, SMOTE = 0.6) | 0.4215 | 73.45% | 82.81% | Lớp 0 <br> Lớp 1 | 88.26% <br> 53.78% | 71.72% <br> 77.52% | 79.14% <br> 63.51% | 1007 | 397 | 134 | 462 |
| MI Top 5 (Cố định, SMOTE = 0.7) | 0.4110 | 72.35% | 82.68% | Lớp 0 <br> Lớp 1 | 88.44% <br> 52.41% | 69.73% <br> 78.52% | 77.98% <br> 62.86% | 979 | 425 | 128 | 468 |
| MI Top 5 (Động, SMOTE = 0.5) | 0.3920 | 72.90% | 82.77% | Lớp 0 <br> Lớp 1 | 88.69% <br> 53.05% | 70.37% <br> 78.86% | 78.47% <br> 63.43% | 988 | 416 | 126 | 470 |
| MI Top 5 (Động, SMOTE = 0.6) | 0.4140 | 73.45% | 82.81% | Lớp 0 <br> Lớp 1 | 88.66% <br> 53.73% | 71.30% <br> 78.52% | 79.04% <br> 63.80% | 1001 | 403 | 128 | 468 |
| MI Top 5 (Động, SMOTE = 0.7) | 0.3995 | 71.65% | 82.68% | Lớp 0 <br> Lớp 1 | 88.64% <br> 51.58% | 68.38% <br> 79.36% | 77.20% <br> 62.52% | 960 | 444 | 123 | 473 |

### ❖ XGBoost (Mã cấu hình cốt lõi: `XGB-C3` | Tham số: `n_estimators=150, max_depth=4, learning_rate=0.1`)

| Kịch bản | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Full 11 (Không SMOTE) | 0.2300 | 73.25% | 83.80% | Lớp 0 <br> Lớp 1 | 89.83% <br> 53.36% | 69.80% <br> 81.38% | 78.56% <br> 64.45% | 980 | 424 | 111 | 485 |
| Full 11 (SMOTE = 0.5) | 0.2335 | 70.85% | 83.55% | Lớp 0 <br> Lớp 1 | 90.13% <br> 50.67% | 65.67% <br> 83.05% | 75.98% <br> 62.94% | 922 | 482 | 101 | 495 |
| Full 11 (SMOTE = 0.6) | 0.2765 | 71.90% | 83.69% | Lớp 0 <br> Lớp 1 | 89.94% <br> 51.80% | 67.52% <br> 82.21% | 77.14% <br> 63.55% | 948 | 456 | 106 | 490 |
| Full 11 (SMOTE = 0.7) | 0.3270 | 73.15% | 84.08% | Lớp 0 <br> Lớp 1 | 89.59% <br> 53.26% | 69.87% <br> 80.87% | 78.51% <br> 64.22% | 981 | 423 | 114 | 482 |
| MI Top 5 (Cố định, SMOTE = 0.5) | 0.2335 | 70.05% | 83.17% | Lớp 0 <br> Lớp 1 | 89.73% <br> 49.85% | 64.74% <br> 82.55% | 75.22% <br> 62.16% | 909 | 495 | 104 | 492 |
| MI Top 5 (Cố định, SMOTE = 0.6) | 0.2765 | 70.65% | 82.91% | Lớp 0 <br> Lớp 1 | 88.43% <br> 50.48% | 66.95% <br> 79.36% | 76.21% <br> 61.71% | 940 | 464 | 123 | 473 |
| MI Top 5 (Cố định, SMOTE = 0.7) | 0.3270 | 72.40% | 82.99% | Lớp 0 <br> Lớp 1 | 88.38% <br> 52.47% | 69.87% <br> 78.36% | 78.04% <br> 62.85% | 981 | 423 | 129 | 467 |
| MI Top 5 (Động, SMOTE = 0.5) | 0.2510 | 71.80% | 83.17% | Lớp 0 <br> Lớp 1 | 88.67% <br> 51.75% | 68.59% <br> 79.36% | 77.35% <br> 62.65% | 963 | 441 | 123 | 473 |
| MI Top 5 (Động, SMOTE = 0.6) | 0.2735 | 70.85% | 82.91% | Lớp 0 <br> Lớp 1 | 88.91% <br> 50.69% | 66.81% <br> 80.37% | 76.29% <br> 62.17% | 938 | 466 | 117 | 479 |
| MI Top 5 (Động, SMOTE = 0.7) | 0.3130 | 71.45% | 82.99% | Lớp 0 <br> Lớp 1 | 88.96% <br> 51.34% | 67.74% <br> 80.20% | 76.91% <br> 62.61% | 951 | 453 | 118 | 478 |

### ❖ Decision Tree (Mã cấu hình cốt lõi: `DT-C3` | Tham số: `max_depth=6, min_samples_leaf=20, class_weight='balanced'`)

| Kịch bản | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Full 11 (Không SMOTE) | 0.3860 | 71.00% | 83.74% | Lớp 0 <br> Lớp 1 | 89.92% <br> 50.83% | 66.10% <br> 82.55% | 76.19% <br> 62.92% | 928 | 476 | 104 | 492 |
| Full 11 (SMOTE = 0.5) | 0.4105 | 70.80% | 83.51% | Lớp 0 <br> Lớp 1 | 89.81% <br> 50.62% | 65.88% <br> 82.38% | 76.01% <br> 62.71% | 925 | 479 | 105 | 491 |
| Full 11 (SMOTE = 0.6) | 0.4090 | 71.40% | 83.43% | Lớp 0 <br> Lớp 1 | 89.62% <br> 51.26% | 67.02% <br> 81.71% | 76.69% <br> 63.00% | 941 | 463 | 109 | 487 |
| Full 11 (SMOTE = 0.7) | 0.4325 | 72.25% | 83.15% | Lớp 0 <br> Lớp 1 | 89.20% <br> 52.24% | 68.80% <br> 80.37% | 77.68% <br> 63.32% | 966 | 438 | 117 | 479 |
| MI Top 5 (Cố định, SMOTE = 0.5) | 0.4105 | 71.65% | 82.68% | Lớp 0 <br> Lớp 1 | 89.44% <br> 51.54% | 67.59% <br> 81.21% | 77.00% <br> 63.06% | 949 | 455 | 112 | 484 |
| MI Top 5 (Cố định, SMOTE = 0.6) | 0.4090 | 72.95% | 82.51% | Lớp 0 <br> Lớp 1 | 88.70% <br> 53.11% | 70.44% <br> 78.86% | 78.52% <br> 63.47% | 989 | 415 | 126 | 470 |
| MI Top 5 (Cố định, SMOTE = 0.7) | 0.4325 | 72.55% | 82.79% | Lớp 0 <br> Lớp 1 | 88.69% <br> 52.63% | 69.80% <br> 79.03% | 78.12% <br> 63.18% | 980 | 424 | 125 | 471 |
| MI Top 5 (Động, SMOTE = 0.5) | 0.4315 | 73.15% | 82.68% | Lớp 0 <br> Lớp 1 | 89.30% <br> 53.29% | 70.16% <br> 80.20% | 78.58% <br> 64.03% | 985 | 419 | 118 | 478 |
| MI Top 5 (Động, SMOTE = 0.6) | 0.3425 | 69.05% | 82.51% | Lớp 0 <br> Lớp 1 | 89.53% <br> 48.86% | 63.32% <br> 82.55% | 74.18% <br> 61.38% | 889 | 515 | 104 | 492 |
| MI Top 5 (Động, SMOTE = 0.7) | 0.4495 | 73.00% | 82.79% | Lớp 0 <br> Lớp 1 | 88.50% <br> 53.19% | 70.73% <br> 78.36% | 78.62% <br> 63.36% | 993 | 411 | 129 | 467 |

## 📝 2. BẢNG CHI TIẾT TOÀN BỘ CÁC CẤU HÌNH THEO TỪNG THUẬT TOÁN VÀ MỐC SMOTE

### ❖ Thuật toán: Logistic Regression

#### A. Không gian Full 11 Features (Không SMOTE)

**Tham số chi tiết của các cấu hình:**
- **LR-C1**: `C=0.01, penalty='l2', class_weight='balanced'`
- **LR-C2**: `C=0.1, penalty='l2', class_weight='balanced'`
- **LR-C3**: `C=1.0, penalty='l2', class_weight='balanced'`
- **LR-C4**: `C=10.0, penalty='l2', class_weight='balanced'`
- **LR-C5**: `C=100.0, penalty='l2', class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LR-C1 | 0.4655 | 73.80% | 84.43% | Lớp 0 <br> Lớp 1 | 89.01% <br> 54.13% | 71.51% <br> 79.19% | 79.30% <br> 64.31% | 1004 | 400 | 124 | 472 |
| LR-C2 | 0.4530 | 74.25% | 85.07% | Lớp 0 <br> Lớp 1 | 89.37% <br> 54.65% | 71.87% <br> 79.87% | 79.67% <br> 64.89% | 1009 | 395 | 120 | 476 |
| LR-C3 | 0.4445 | 74.00% | 85.10% | Lớp 0 <br> Lớp 1 | 89.61% <br> 54.30% | 71.23% <br> 80.54% | 79.37% <br> 64.86% | 1000 | 404 | 116 | 480 |
| LR-C4 | 0.4440 | 73.95% | 85.11% | Lớp 0 <br> Lớp 1 | 89.60% <br> 54.24% | 71.15% <br> 80.54% | 79.32% <br> 64.82% | 999 | 405 | 116 | 480 |
| LR-C5 | 0.4440 | 73.95% | 85.11% | Lớp 0 <br> Lớp 1 | 89.60% <br> 54.24% | 71.15% <br> 80.54% | 79.32% <br> 64.82% | 999 | 405 | 116 | 480 |

#### B. Không gian Full 11 Features (Có SMOTE = 0.5)

**Tham số chi tiết của các cấu hình:**
- **LR-C1**: `C=0.01, penalty='l2', class_weight='balanced'`
- **LR-C2**: `C=0.1, penalty='l2', class_weight='balanced'`
- **LR-C3**: `C=1.0, penalty='l2', class_weight='balanced'`
- **LR-C4**: `C=10.0, penalty='l2', class_weight='balanced'`
- **LR-C5**: `C=100.0, penalty='l2', class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LR-C1 | 0.4705 | 73.95% | 84.39% | Lớp 0 <br> Lớp 1 | 89.11% <br> 54.31% | 71.65% <br> 79.36% | 79.43% <br> 64.49% | 1006 | 398 | 123 | 473 |
| LR-C2 | 0.4575 | 74.20% | 85.02% | Lớp 0 <br> Lớp 1 | 89.50% <br> 54.57% | 71.65% <br> 80.20% | 79.59% <br> 64.95% | 1006 | 398 | 118 | 478 |
| LR-C3 | 0.4465 | 73.90% | 85.06% | Lớp 0 <br> Lớp 1 | 89.80% <br> 54.15% | 70.87% <br> 81.04% | 79.22% <br> 64.92% | 995 | 409 | 113 | 483 |
| LR-C4 | 0.4455 | 73.90% | 85.07% | Lớp 0 <br> Lớp 1 | 89.80% <br> 54.15% | 70.87% <br> 81.04% | 79.22% <br> 64.92% | 995 | 409 | 113 | 483 |
| LR-C5 | 0.4455 | 73.90% | 85.08% | Lớp 0 <br> Lớp 1 | 89.80% <br> 54.15% | 70.87% <br> 81.04% | 79.22% <br> 64.92% | 995 | 409 | 113 | 483 |

#### C. Không gian Full 11 Features (Có SMOTE = 0.6)

**Tham số chi tiết của các cấu hình:**
- **LR-C1**: `C=0.01, penalty='l2', class_weight='balanced'`
- **LR-C2**: `C=0.1, penalty='l2', class_weight='balanced'`
- **LR-C3**: `C=1.0, penalty='l2', class_weight='balanced'`
- **LR-C4**: `C=10.0, penalty='l2', class_weight='balanced'`
- **LR-C5**: `C=100.0, penalty='l2', class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LR-C1 | 0.4725 | 73.95% | 84.40% | Lớp 0 <br> Lớp 1 | 89.17% <br> 54.30% | 71.58% <br> 79.53% | 79.42% <br> 64.53% | 1005 | 399 | 122 | 474 |
| LR-C2 | 0.4615 | 74.20% | 85.00% | Lớp 0 <br> Lớp 1 | 89.36% <br> 54.59% | 71.79% <br> 79.87% | 79.62% <br> 64.85% | 1008 | 396 | 120 | 476 |
| LR-C3 | 0.4575 | 74.25% | 85.05% | Lớp 0 <br> Lớp 1 | 89.58% <br> 54.62% | 71.65% <br> 80.37% | 79.62% <br> 65.04% | 1006 | 398 | 117 | 479 |
| LR-C4 | 0.4490 | 74.05% | 85.07% | Lớp 0 <br> Lớp 1 | 89.83% <br> 54.33% | 71.08% <br> 81.04% | 79.36% <br> 65.05% | 998 | 406 | 113 | 483 |
| LR-C5 | 0.4555 | 74.25% | 85.07% | Lớp 0 <br> Lớp 1 | 89.58% <br> 54.62% | 71.65% <br> 80.37% | 79.62% <br> 65.04% | 1006 | 398 | 117 | 479 |

#### D. Không gian Full 11 Features (Có SMOTE = 0.7)

**Tham số chi tiết của các cấu hình:**
- **LR-C1**: `C=0.01, penalty='l2', class_weight='balanced'`
- **LR-C2**: `C=0.1, penalty='l2', class_weight='balanced'`
- **LR-C3**: `C=1.0, penalty='l2', class_weight='balanced'`
- **LR-C4**: `C=10.0, penalty='l2', class_weight='balanced'`
- **LR-C5**: `C=100.0, penalty='l2', class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LR-C1 | 0.4690 | 73.65% | 84.37% | Lớp 0 <br> Lớp 1 | 89.26% <br> 53.91% | 71.01% <br> 79.87% | 79.10% <br> 64.37% | 997 | 407 | 120 | 476 |
| LR-C2 | 0.4715 | 74.50% | 85.01% | Lớp 0 <br> Lớp 1 | 89.21% <br> 55.00% | 72.44% <br> 79.36% | 79.95% <br> 64.97% | 1017 | 387 | 123 | 473 |
| LR-C3 | 0.4685 | 74.55% | 85.06% | Lớp 0 <br> Lớp 1 | 89.22% <br> 55.06% | 72.51% <br> 79.36% | 80.00% <br> 65.02% | 1018 | 386 | 123 | 473 |
| LR-C4 | 0.4670 | 74.50% | 85.07% | Lớp 0 <br> Lớp 1 | 89.21% <br> 55.00% | 72.44% <br> 79.36% | 79.95% <br> 64.97% | 1017 | 387 | 123 | 473 |
| LR-C5 | 0.4675 | 74.55% | 85.07% | Lớp 0 <br> Lớp 1 | 89.22% <br> 55.06% | 72.51% <br> 79.36% | 80.00% <br> 65.02% | 1018 | 386 | 123 | 473 |

#### E. Không gian MI Top 5 Features (Có SMOTE = 0.5) - Ngưỡng động tối ưu

**Tham số chi tiết của các cấu hình:**
- **LR-C1**: `C=0.01, penalty='l2', class_weight='balanced'`
- **LR-C2**: `C=0.1, penalty='l2', class_weight='balanced'`
- **LR-C3**: `C=1.0, penalty='l2', class_weight='balanced'`
- **LR-C4**: `C=10.0, penalty='l2', class_weight='balanced'`
- **LR-C5**: `C=100.0, penalty='l2', class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LR-C1 | 0.4660 | 72.95% | 83.95% | Lớp 0 <br> Lớp 1 | 89.55% <br> 53.03% | 69.59% <br> 80.87% | 78.32% <br> 64.05% | 977 | 427 | 114 | 482 |
| LR-C2 | 0.4560 | 74.05% | 84.57% | Lớp 0 <br> Lớp 1 | 90.34% <br> 54.26% | 70.58% <br> 82.21% | 79.25% <br> 65.38% | 991 | 413 | 106 | 490 |
| LR-C3 | 0.4425 | 73.50% | 84.63% | Lớp 0 <br> Lớp 1 | 90.39% <br> 53.59% | 69.66% <br> 82.55% | 78.68% <br> 64.99% | 978 | 426 | 104 | 492 |
| LR-C4 | 0.4480 | 74.00% | 84.63% | Lớp 0 <br> Lớp 1 | 90.48% <br> 54.19% | 70.37% <br> 82.55% | 79.17% <br> 65.43% | 988 | 416 | 104 | 492 |
| LR-C5 | 0.4480 | 74.00% | 84.63% | Lớp 0 <br> Lớp 1 | 90.48% <br> 54.19% | 70.37% <br> 82.55% | 79.17% <br> 65.43% | 988 | 416 | 104 | 492 |

#### F. Không gian MI Top 5 Features (Có SMOTE = 0.6) - Ngưỡng động tối ưu

**Tham số chi tiết của các cấu hình:**
- **LR-C1**: `C=0.01, penalty='l2', class_weight='balanced'`
- **LR-C2**: `C=0.1, penalty='l2', class_weight='balanced'`
- **LR-C3**: `C=1.0, penalty='l2', class_weight='balanced'`
- **LR-C4**: `C=10.0, penalty='l2', class_weight='balanced'`
- **LR-C5**: `C=100.0, penalty='l2', class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LR-C1 | 0.4750 | 73.55% | 83.97% | Lớp 0 <br> Lớp 1 | 89.17% <br> 53.79% | 70.94% <br> 79.70% | 79.02% <br> 64.23% | 996 | 408 | 121 | 475 |
| LR-C2 | 0.4565 | 74.05% | 84.58% | Lớp 0 <br> Lớp 1 | 90.34% <br> 54.26% | 70.58% <br> 82.21% | 79.25% <br> 65.38% | 991 | 413 | 106 | 490 |
| LR-C3 | 0.4505 | 73.95% | 84.63% | Lớp 0 <br> Lớp 1 | 90.47% <br> 54.13% | 70.30% <br> 82.55% | 79.12% <br> 65.38% | 987 | 417 | 104 | 492 |
| LR-C4 | 0.4430 | 73.45% | 84.64% | Lớp 0 <br> Lớp 1 | 90.45% <br> 53.53% | 69.52% <br> 82.72% | 78.61% <br> 65.00% | 976 | 428 | 103 | 493 |
| LR-C5 | 0.4430 | 73.45% | 84.64% | Lớp 0 <br> Lớp 1 | 90.45% <br> 53.53% | 69.52% <br> 82.72% | 78.61% <br> 65.00% | 976 | 428 | 103 | 493 |

#### G. Không gian MI Top 5 Features (Có SMOTE = 0.7) - Ngưỡng động tối ưu

**Tham số chi tiết của các cấu hình:**
- **LR-C1**: `C=0.01, penalty='l2', class_weight='balanced'`
- **LR-C2**: `C=0.1, penalty='l2', class_weight='balanced'`
- **LR-C3**: `C=1.0, penalty='l2', class_weight='balanced'`
- **LR-C4**: `C=10.0, penalty='l2', class_weight='balanced'`
- **LR-C5**: `C=100.0, penalty='l2', class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LR-C1 | 0.4790 | 73.95% | 83.93% | Lớp 0 <br> Lớp 1 | 89.31% <br> 54.28% | 71.44% <br> 79.87% | 79.38% <br> 64.63% | 1003 | 401 | 120 | 476 |
| LR-C2 | 0.4640 | 74.40% | 84.57% | Lớp 0 <br> Lớp 1 | 90.25% <br> 54.71% | 71.23% <br> 81.88% | 79.62% <br> 65.59% | 1000 | 404 | 108 | 488 |
| LR-C3 | 0.4550 | 73.90% | 84.62% | Lớp 0 <br> Lớp 1 | 90.31% <br> 54.08% | 70.37% <br> 82.21% | 79.10% <br> 65.25% | 988 | 416 | 106 | 490 |
| LR-C4 | 0.4550 | 73.90% | 84.62% | Lớp 0 <br> Lớp 1 | 90.31% <br> 54.08% | 70.37% <br> 82.21% | 79.10% <br> 65.25% | 988 | 416 | 106 | 490 |
| LR-C5 | 0.4550 | 73.90% | 84.62% | Lớp 0 <br> Lớp 1 | 90.31% <br> 54.08% | 70.37% <br> 82.21% | 79.10% <br> 65.25% | 988 | 416 | 106 | 490 |

### ❖ Thuật toán: SVM (RBF)

#### A. Không gian Full 11 Features (Không SMOTE)

**Tham số chi tiết của các cấu hình:**
- **SVM-C1**: `C=0.1, kernel='rbf', gamma='scale', class_weight='balanced'`
- **SVM-C2**: `C=1.0, kernel='rbf', gamma='scale', class_weight='balanced'`
- **SVM-C3**: `C=10.0, kernel='rbf', gamma=0.01, class_weight='balanced'`
- **SVM-C4**: `C=10.0, kernel='rbf', gamma=0.1, class_weight='balanced'`
- **SVM-C5**: `C=100.0, kernel='rbf', gamma=0.01, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SVM-C1 | 0.2095 | 72.65% | 84.26% | Lớp 0 <br> Lớp 1 | 89.86% <br> 52.65% | 68.80% <br> 81.71% | 77.93% <br> 64.04% | 966 | 438 | 109 | 487 |
| SVM-C2 | 0.1770 | 74.00% | 84.20% | Lớp 0 <br> Lớp 1 | 90.11% <br> 54.23% | 70.73% <br> 81.71% | 79.25% <br> 65.19% | 993 | 411 | 109 | 487 |
| SVM-C3 | 0.2300 | 74.45% | 85.08% | Lớp 0 <br> Lớp 1 | 89.62% <br> 54.87% | 71.94% <br> 80.37% | 79.81% <br> 65.21% | 1010 | 394 | 117 | 479 |
| SVM-C4 | 0.1715 | 74.25% | 83.88% | Lớp 0 <br> Lớp 1 | 90.08% <br> 54.55% | 71.15% <br> 81.54% | 79.51% <br> 65.37% | 999 | 405 | 110 | 486 |
| SVM-C5 | 0.2040 | 73.85% | 84.77% | Lớp 0 <br> Lớp 1 | 89.65% <br> 54.11% | 70.94% <br> 80.70% | 79.20% <br> 64.78% | 996 | 408 | 115 | 481 |

#### B. Không gian Full 11 Features (Có SMOTE = 0.5)

**Tham số chi tiết của các cấu hình:**
- **SVM-C1**: `C=0.1, kernel='rbf', gamma='scale', class_weight='balanced'`
- **SVM-C2**: `C=1.0, kernel='rbf', gamma='scale', class_weight='balanced'`
- **SVM-C3**: `C=10.0, kernel='rbf', gamma=0.01, class_weight='balanced'`
- **SVM-C4**: `C=10.0, kernel='rbf', gamma=0.1, class_weight='balanced'`
- **SVM-C5**: `C=100.0, kernel='rbf', gamma=0.01, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SVM-C1 | 0.2685 | 73.90% | 84.19% | Lớp 0 <br> Lớp 1 | 88.96% <br> 54.26% | 71.72% <br> 79.03% | 79.42% <br> 64.34% | 1007 | 397 | 125 | 471 |
| SVM-C2 | 0.1840 | 73.30% | 84.00% | Lớp 0 <br> Lớp 1 | 90.28% <br> 53.37% | 69.44% <br> 82.38% | 78.50% <br> 64.78% | 975 | 429 | 105 | 491 |
| SVM-C3 | 0.2720 | 73.70% | 85.09% | Lớp 0 <br> Lớp 1 | 89.34% <br> 53.96% | 71.01% <br> 80.03% | 79.13% <br> 64.46% | 997 | 407 | 119 | 477 |
| SVM-C4 | 0.1965 | 74.25% | 83.65% | Lớp 0 <br> Lớp 1 | 89.65% <br> 54.61% | 71.58% <br> 80.54% | 79.60% <br> 65.08% | 1005 | 399 | 116 | 480 |
| SVM-C5 | 0.1765 | 70.05% | 84.64% | Lớp 0 <br> Lớp 1 | 91.20% <br> 49.85% | 63.46% <br> 85.57% | 74.84% <br> 63.00% | 891 | 513 | 86 | 510 |

#### C. Không gian Full 11 Features (Có SMOTE = 0.6)

**Tham số chi tiết của các cấu hình:**
- **SVM-C1**: `C=0.1, kernel='rbf', gamma='scale', class_weight='balanced'`
- **SVM-C2**: `C=1.0, kernel='rbf', gamma='scale', class_weight='balanced'`
- **SVM-C3**: `C=10.0, kernel='rbf', gamma=0.01, class_weight='balanced'`
- **SVM-C4**: `C=10.0, kernel='rbf', gamma=0.1, class_weight='balanced'`
- **SVM-C5**: `C=100.0, kernel='rbf', gamma=0.01, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SVM-C1 | 0.2520 | 71.80% | 84.17% | Lớp 0 <br> Lớp 1 | 90.38% <br> 51.67% | 66.95% <br> 83.22% | 76.92% <br> 63.75% | 940 | 464 | 100 | 496 |
| SVM-C2 | 0.2100 | 73.35% | 83.83% | Lớp 0 <br> Lớp 1 | 90.51% <br> 53.41% | 69.30% <br> 82.89% | 78.50% <br> 64.96% | 973 | 431 | 102 | 494 |
| SVM-C3 | 0.3215 | 74.25% | 85.07% | Lớp 0 <br> Lớp 1 | 89.16% <br> 54.68% | 72.08% <br> 79.36% | 79.72% <br> 64.75% | 1012 | 392 | 123 | 473 |
| SVM-C4 | 0.2140 | 73.90% | 83.40% | Lớp 0 <br> Lớp 1 | 90.16% <br> 54.10% | 70.51% <br> 81.88% | 79.14% <br> 65.15% | 990 | 414 | 108 | 488 |
| SVM-C5 | 0.2530 | 73.30% | 84.61% | Lớp 0 <br> Lớp 1 | 89.62% <br> 53.44% | 70.09% <br> 80.87% | 78.66% <br> 64.35% | 984 | 420 | 114 | 482 |

#### D. Không gian Full 11 Features (Có SMOTE = 0.7)

**Tham số chi tiết của các cấu hình:**
- **SVM-C1**: `C=0.1, kernel='rbf', gamma='scale', class_weight='balanced'`
- **SVM-C2**: `C=1.0, kernel='rbf', gamma='scale', class_weight='balanced'`
- **SVM-C3**: `C=10.0, kernel='rbf', gamma=0.01, class_weight='balanced'`
- **SVM-C4**: `C=10.0, kernel='rbf', gamma=0.1, class_weight='balanced'`
- **SVM-C5**: `C=100.0, kernel='rbf', gamma=0.01, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SVM-C1 | 0.3510 | 74.20% | 84.11% | Lớp 0 <br> Lớp 1 | 89.08% <br> 54.63% | 72.08% <br> 79.19% | 79.69% <br> 64.66% | 1012 | 392 | 124 | 472 |
| SVM-C2 | 0.2445 | 73.75% | 83.97% | Lớp 0 <br> Lớp 1 | 90.58% <br> 53.87% | 69.87% <br> 82.89% | 78.89% <br> 65.30% | 981 | 423 | 102 | 494 |
| SVM-C3 | 0.3425 | 73.15% | 84.97% | Lớp 0 <br> Lớp 1 | 89.59% <br> 53.26% | 69.87% <br> 80.87% | 78.51% <br> 64.22% | 981 | 423 | 114 | 482 |
| SVM-C4 | 0.2425 | 73.70% | 83.43% | Lớp 0 <br> Lớp 1 | 89.98% <br> 53.88% | 70.37% <br> 81.54% | 78.98% <br> 64.89% | 988 | 416 | 110 | 486 |
| SVM-C5 | 0.2890 | 72.75% | 84.59% | Lớp 0 <br> Lớp 1 | 89.37% <br> 52.81% | 69.44% <br> 80.54% | 78.16% <br> 63.79% | 975 | 429 | 116 | 480 |

#### E. Không gian MI Top 5 Features (Có SMOTE = 0.5) - Ngưỡng động tối ưu

**Tham số chi tiết của các cấu hình:**
- **SVM-C1**: `C=0.1, kernel='rbf', gamma='scale', class_weight='balanced'`
- **SVM-C2**: `C=1.0, kernel='rbf', gamma='scale', class_weight='balanced'`
- **SVM-C3**: `C=10.0, kernel='rbf', gamma=0.01, class_weight='balanced'`
- **SVM-C4**: `C=10.0, kernel='rbf', gamma=0.1, class_weight='balanced'`
- **SVM-C5**: `C=100.0, kernel='rbf', gamma=0.01, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SVM-C1 | 0.1700 | 72.25% | 83.35% | Lớp 0 <br> Lớp 1 | 90.31% <br> 52.16% | 67.74% <br> 82.89% | 77.41% <br> 64.03% | 951 | 453 | 102 | 494 |
| SVM-C2 | 0.1560 | 74.30% | 82.63% | Lớp 0 <br> Lớp 1 | 90.24% <br> 54.59% | 71.08% <br> 81.88% | 79.52% <br> 65.50% | 998 | 406 | 108 | 488 |
| SVM-C3 | 0.2640 | 74.90% | 84.51% | Lớp 0 <br> Lớp 1 | 90.05% <br> 55.38% | 72.22% <br> 81.21% | 80.16% <br> 65.85% | 1014 | 390 | 112 | 484 |
| SVM-C4 | 0.1665 | 73.90% | 83.03% | Lớp 0 <br> Lớp 1 | 90.46% <br> 54.07% | 70.23% <br> 82.55% | 79.07% <br> 65.34% | 986 | 418 | 104 | 492 |
| SVM-C5 | 0.2155 | 74.55% | 83.98% | Lớp 0 <br> Lớp 1 | 89.71% <br> 54.98% | 72.01% <br> 80.54% | 79.89% <br> 65.35% | 1011 | 393 | 116 | 480 |

#### F. Không gian MI Top 5 Features (Có SMOTE = 0.6) - Ngưỡng động tối ưu

**Tham số chi tiết của các cấu hình:**
- **SVM-C1**: `C=0.1, kernel='rbf', gamma='scale', class_weight='balanced'`
- **SVM-C2**: `C=1.0, kernel='rbf', gamma='scale', class_weight='balanced'`
- **SVM-C3**: `C=10.0, kernel='rbf', gamma=0.01, class_weight='balanced'`
- **SVM-C4**: `C=10.0, kernel='rbf', gamma=0.1, class_weight='balanced'`
- **SVM-C5**: `C=100.0, kernel='rbf', gamma=0.01, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SVM-C1 | 0.1995 | 72.85% | 83.26% | Lớp 0 <br> Lớp 1 | 90.12% <br> 52.86% | 68.87% <br> 82.21% | 78.08% <br> 64.35% | 967 | 437 | 106 | 490 |
| SVM-C2 | 0.1775 | 74.05% | 82.76% | Lớp 0 <br> Lớp 1 | 90.26% <br> 54.27% | 70.66% <br> 82.05% | 79.26% <br> 65.33% | 992 | 412 | 107 | 489 |
| SVM-C3 | 0.3090 | 74.95% | 84.53% | Lớp 0 <br> Lớp 1 | 89.85% <br> 55.48% | 72.51% <br> 80.70% | 80.25% <br> 65.76% | 1018 | 386 | 115 | 481 |
| SVM-C4 | 0.1895 | 73.85% | 82.95% | Lớp 0 <br> Lớp 1 | 90.52% <br> 54.00% | 70.09% <br> 82.72% | 79.00% <br> 65.34% | 984 | 420 | 103 | 493 |
| SVM-C5 | 0.2245 | 73.70% | 83.86% | Lớp 0 <br> Lớp 1 | 89.27% <br> 53.97% | 71.08% <br> 79.87% | 79.14% <br> 64.41% | 998 | 406 | 120 | 476 |

#### G. Không gian MI Top 5 Features (Có SMOTE = 0.7) - Ngưỡng động tối ưu

**Tham số chi tiết của các cấu hình:**
- **SVM-C1**: `C=0.1, kernel='rbf', gamma='scale', class_weight='balanced'`
- **SVM-C2**: `C=1.0, kernel='rbf', gamma='scale', class_weight='balanced'`
- **SVM-C3**: `C=10.0, kernel='rbf', gamma=0.01, class_weight='balanced'`
- **SVM-C4**: `C=10.0, kernel='rbf', gamma=0.1, class_weight='balanced'`
- **SVM-C5**: `C=100.0, kernel='rbf', gamma=0.01, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SVM-C1 | 0.2335 | 73.30% | 83.17% | Lớp 0 <br> Lớp 1 | 90.20% <br> 53.38% | 69.52% <br> 82.21% | 78.52% <br> 64.73% | 976 | 428 | 106 | 490 |
| SVM-C2 | 0.1905 | 72.40% | 82.82% | Lớp 0 <br> Lớp 1 | 90.96% <br> 52.29% | 67.38% <br> 84.23% | 77.41% <br> 64.52% | 946 | 458 | 94 | 502 |
| SVM-C3 | 0.3625 | 74.10% | 84.42% | Lớp 0 <br> Lớp 1 | 90.13% <br> 54.35% | 70.87% <br> 81.71% | 79.35% <br> 65.28% | 995 | 409 | 109 | 487 |
| SVM-C4 | 0.2270 | 73.90% | 83.06% | Lớp 0 <br> Lớp 1 | 90.46% <br> 54.07% | 70.23% <br> 82.55% | 79.07% <br> 65.34% | 986 | 418 | 104 | 492 |
| SVM-C5 | 0.2565 | 72.60% | 83.87% | Lớp 0 <br> Lớp 1 | 89.85% <br> 52.59% | 68.73% <br> 81.71% | 77.89% <br> 63.99% | 965 | 439 | 109 | 487 |

### ❖ Thuật toán: Random Forest

#### A. Không gian Full 11 Features (Không SMOTE)

**Tham số chi tiết của các cấu hình:**
- **RF-C1**: `n_estimators=50, max_depth=4, class_weight='balanced'`
- **RF-C2**: `n_estimators=100, max_depth=6, class_weight='balanced'`
- **RF-C3**: `n_estimators=200, max_depth=8, class_weight='balanced'`
- **RF-C4**: `n_estimators=300, max_depth=8, min_samples_split=10, class_weight='balanced'`
- **RF-C5**: `n_estimators=500, max_depth=12, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RF-C1 | 0.4980 | 75.25% | 84.23% | Lớp 0 <br> Lớp 1 | 89.01% <br> 56.05% | 73.86% <br> 78.52% | 80.73% <br> 65.41% | 1037 | 367 | 128 | 468 |
| RF-C2 | 0.4525 | 74.20% | 84.71% | Lớp 0 <br> Lớp 1 | 89.57% <br> 54.56% | 71.58% <br> 80.37% | 79.57% <br> 64.99% | 1005 | 399 | 117 | 479 |
| RF-C3 | 0.4145 | 73.50% | 84.65% | Lớp 0 <br> Lớp 1 | 89.58% <br> 53.68% | 70.44% <br> 80.70% | 78.87% <br> 64.48% | 989 | 415 | 115 | 481 |
| RF-C4 | 0.4190 | 73.40% | 84.67% | Lớp 0 <br> Lớp 1 | 89.21% <br> 53.60% | 70.66% <br> 79.87% | 78.86% <br> 64.15% | 992 | 412 | 120 | 476 |
| RF-C5 | 0.3445 | 72.60% | 83.77% | Lớp 0 <br> Lớp 1 | 88.42% <br> 52.71% | 70.16% <br> 78.36% | 78.24% <br> 63.02% | 985 | 419 | 129 | 467 |

#### B. Không gian Full 11 Features (Có SMOTE = 0.5)

**Tham số chi tiết của các cấu hình:**
- **RF-C1**: `n_estimators=50, max_depth=4, class_weight='balanced'`
- **RF-C2**: `n_estimators=100, max_depth=6, class_weight='balanced'`
- **RF-C3**: `n_estimators=200, max_depth=8, class_weight='balanced'`
- **RF-C4**: `n_estimators=300, max_depth=8, min_samples_split=10, class_weight='balanced'`
- **RF-C5**: `n_estimators=500, max_depth=12, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RF-C1 | 0.4710 | 73.80% | 84.31% | Lớp 0 <br> Lớp 1 | 89.86% <br> 54.02% | 70.66% <br> 81.21% | 79.11% <br> 64.88% | 992 | 412 | 112 | 484 |
| RF-C2 | 0.4430 | 73.00% | 84.71% | Lớp 0 <br> Lớp 1 | 90.30% <br> 53.02% | 68.95% <br> 82.55% | 78.19% <br> 64.57% | 968 | 436 | 104 | 492 |
| RF-C3 | 0.3875 | 71.05% | 84.50% | Lớp 0 <br> Lớp 1 | 90.64% <br> 50.86% | 65.53% <br> 84.06% | 76.06% <br> 63.38% | 920 | 484 | 95 | 501 |
| RF-C4 | 0.4145 | 73.25% | 84.51% | Lớp 0 <br> Lớp 1 | 89.39% <br> 53.40% | 70.23% <br> 80.37% | 78.66% <br> 64.17% | 986 | 418 | 117 | 479 |
| RF-C5 | 0.3505 | 72.00% | 83.63% | Lớp 0 <br> Lớp 1 | 88.22% <br> 52.01% | 69.37% <br> 78.19% | 77.67% <br> 62.47% | 974 | 430 | 130 | 466 |

#### C. Không gian Full 11 Features (Có SMOTE = 0.6)

**Tham số chi tiết của các cấu hình:**
- **RF-C1**: `n_estimators=50, max_depth=4, class_weight='balanced'`
- **RF-C2**: `n_estimators=100, max_depth=6, class_weight='balanced'`
- **RF-C3**: `n_estimators=200, max_depth=8, class_weight='balanced'`
- **RF-C4**: `n_estimators=300, max_depth=8, min_samples_split=10, class_weight='balanced'`
- **RF-C5**: `n_estimators=500, max_depth=12, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RF-C1 | 0.4935 | 74.40% | 84.25% | Lớp 0 <br> Lớp 1 | 88.99% <br> 54.91% | 72.51% <br> 78.86% | 79.91% <br> 64.74% | 1018 | 386 | 126 | 470 |
| RF-C2 | 0.4515 | 74.15% | 84.82% | Lớp 0 <br> Lớp 1 | 89.92% <br> 54.44% | 71.15% <br> 81.21% | 79.44% <br> 65.19% | 999 | 405 | 112 | 484 |
| RF-C3 | 0.4215 | 73.40% | 84.58% | Lớp 0 <br> Lớp 1 | 89.14% <br> 53.61% | 70.73% <br> 79.70% | 78.87% <br> 64.10% | 993 | 411 | 121 | 475 |
| RF-C4 | 0.4205 | 73.05% | 84.57% | Lớp 0 <br> Lớp 1 | 89.28% <br> 53.17% | 70.01% <br> 80.20% | 78.48% <br> 63.95% | 983 | 421 | 118 | 478 |
| RF-C5 | 0.3540 | 72.05% | 83.64% | Lớp 0 <br> Lớp 1 | 88.66% <br> 52.04% | 69.02% <br> 79.19% | 77.61% <br> 62.81% | 969 | 435 | 124 | 472 |

#### D. Không gian Full 11 Features (Có SMOTE = 0.7)

**Tham số chi tiết của các cấu hình:**
- **RF-C1**: `n_estimators=50, max_depth=4, class_weight='balanced'`
- **RF-C2**: `n_estimators=100, max_depth=6, class_weight='balanced'`
- **RF-C3**: `n_estimators=200, max_depth=8, class_weight='balanced'`
- **RF-C4**: `n_estimators=300, max_depth=8, min_samples_split=10, class_weight='balanced'`
- **RF-C5**: `n_estimators=500, max_depth=12, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RF-C1 | 0.4850 | 74.90% | 84.12% | Lớp 0 <br> Lớp 1 | 89.42% <br> 55.49% | 72.86% <br> 79.70% | 80.30% <br> 65.43% | 1023 | 381 | 121 | 475 |
| RF-C2 | 0.4475 | 73.90% | 84.62% | Lớp 0 <br> Lớp 1 | 89.38% <br> 54.20% | 71.30% <br> 80.03% | 79.32% <br> 64.63% | 1001 | 403 | 119 | 477 |
| RF-C3 | 0.4110 | 72.35% | 84.53% | Lớp 0 <br> Lớp 1 | 89.58% <br> 52.32% | 68.59% <br> 81.21% | 77.69% <br> 63.64% | 963 | 441 | 112 | 484 |
| RF-C4 | 0.4180 | 72.45% | 84.48% | Lớp 0 <br> Lớp 1 | 89.75% <br> 52.43% | 68.59% <br> 81.54% | 77.76% <br> 63.82% | 963 | 441 | 110 | 486 |
| RF-C5 | 0.3760 | 72.40% | 83.58% | Lớp 0 <br> Lớp 1 | 88.45% <br> 52.47% | 69.80% <br> 78.52% | 78.03% <br> 62.90% | 980 | 424 | 128 | 468 |

#### E. Không gian MI Top 5 Features (Có SMOTE = 0.5) - Ngưỡng động tối ưu

**Tham số chi tiết của các cấu hình:**
- **RF-C1**: `n_estimators=50, max_depth=4, class_weight='balanced'`
- **RF-C2**: `n_estimators=100, max_depth=6, class_weight='balanced'`
- **RF-C3**: `n_estimators=200, max_depth=8, class_weight='balanced'`
- **RF-C4**: `n_estimators=300, max_depth=8, min_samples_split=10, class_weight='balanced'`
- **RF-C5**: `n_estimators=500, max_depth=12, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RF-C1 | 0.4380 | 74.00% | 84.12% | Lớp 0 <br> Lớp 1 | 89.96% <br> 54.25% | 70.87% <br> 81.38% | 79.28% <br> 65.10% | 995 | 409 | 111 | 485 |
| RF-C2 | 0.4235 | 73.90% | 83.80% | Lớp 0 <br> Lớp 1 | 88.82% <br> 54.28% | 71.87% <br> 78.69% | 79.45% <br> 64.25% | 1009 | 395 | 127 | 469 |
| RF-C3 | 0.3920 | 72.90% | 82.77% | Lớp 0 <br> Lớp 1 | 88.69% <br> 53.05% | 70.37% <br> 78.86% | 78.47% <br> 63.43% | 988 | 416 | 126 | 470 |
| RF-C4 | 0.4045 | 73.05% | 82.90% | Lớp 0 <br> Lớp 1 | 88.24% <br> 53.28% | 71.08% <br> 77.68% | 78.74% <br> 63.21% | 998 | 406 | 133 | 463 |
| RF-C5 | 0.3240 | 69.25% | 80.61% | Lớp 0 <br> Lớp 1 | 88.64% <br> 49.03% | 64.46% <br> 80.54% | 74.64% <br> 60.95% | 905 | 499 | 116 | 480 |

#### F. Không gian MI Top 5 Features (Có SMOTE = 0.6) - Ngưỡng động tối ưu

**Tham số chi tiết của các cấu hình:**
- **RF-C1**: `n_estimators=50, max_depth=4, class_weight='balanced'`
- **RF-C2**: `n_estimators=100, max_depth=6, class_weight='balanced'`
- **RF-C3**: `n_estimators=200, max_depth=8, class_weight='balanced'`
- **RF-C4**: `n_estimators=300, max_depth=8, min_samples_split=10, class_weight='balanced'`
- **RF-C5**: `n_estimators=500, max_depth=12, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RF-C1 | 0.4410 | 74.20% | 84.04% | Lớp 0 <br> Lớp 1 | 89.93% <br> 54.50% | 71.23% <br> 81.21% | 79.49% <br> 65.23% | 1000 | 404 | 112 | 484 |
| RF-C2 | 0.4265 | 73.75% | 83.66% | Lớp 0 <br> Lớp 1 | 88.86% <br> 54.09% | 71.58% <br> 78.86% | 79.29% <br> 64.16% | 1005 | 399 | 126 | 470 |
| RF-C3 | 0.4140 | 73.45% | 82.81% | Lớp 0 <br> Lớp 1 | 88.66% <br> 53.73% | 71.30% <br> 78.52% | 79.04% <br> 63.80% | 1001 | 403 | 128 | 468 |
| RF-C4 | 0.4000 | 72.75% | 82.91% | Lớp 0 <br> Lớp 1 | 88.80% <br> 52.86% | 70.01% <br> 79.19% | 78.30% <br> 63.40% | 983 | 421 | 124 | 472 |
| RF-C5 | 0.3375 | 69.45% | 80.62% | Lớp 0 <br> Lớp 1 | 88.76% <br> 49.23% | 64.67% <br> 80.70% | 74.82% <br> 61.16% | 908 | 496 | 115 | 481 |

#### G. Không gian MI Top 5 Features (Có SMOTE = 0.7) - Ngưỡng động tối ưu

**Tham số chi tiết của các cấu hình:**
- **RF-C1**: `n_estimators=50, max_depth=4, class_weight='balanced'`
- **RF-C2**: `n_estimators=100, max_depth=6, class_weight='balanced'`
- **RF-C3**: `n_estimators=200, max_depth=8, class_weight='balanced'`
- **RF-C4**: `n_estimators=300, max_depth=8, min_samples_split=10, class_weight='balanced'`
- **RF-C5**: `n_estimators=500, max_depth=12, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RF-C1 | 0.4440 | 74.10% | 83.80% | Lớp 0 <br> Lớp 1 | 89.84% <br> 54.39% | 71.15% <br> 81.04% | 79.41% <br> 65.09% | 999 | 405 | 113 | 483 |
| RF-C2 | 0.4325 | 73.85% | 83.70% | Lớp 0 <br> Lớp 1 | 88.54% <br> 54.26% | 72.08% <br> 78.02% | 79.47% <br> 64.01% | 1012 | 392 | 131 | 465 |
| RF-C3 | 0.3995 | 71.65% | 82.68% | Lớp 0 <br> Lớp 1 | 88.64% <br> 51.58% | 68.38% <br> 79.36% | 77.20% <br> 62.52% | 960 | 444 | 123 | 473 |
| RF-C4 | 0.3710 | 69.95% | 82.81% | Lớp 0 <br> Lớp 1 | 89.48% <br> 49.75% | 64.81% <br> 82.05% | 75.18% <br> 61.94% | 910 | 494 | 107 | 489 |
| RF-C5 | 0.3690 | 70.40% | 80.62% | Lớp 0 <br> Lớp 1 | 87.73% <br> 50.22% | 67.24% <br> 77.85% | 76.13% <br> 61.05% | 944 | 460 | 132 | 464 |

### ❖ Thuật toán: XGBoost

#### A. Không gian Full 11 Features (Không SMOTE)

**Tham số chi tiết của các cấu hình:**
- **XGB-C1**: `n_estimators=100, max_depth=3, learning_rate=0.1`
- **XGB-C2**: `n_estimators=200, max_depth=5, learning_rate=0.05`
- **XGB-C3**: `n_estimators=150, max_depth=4, learning_rate=0.1`
- **XGB-C4**: `n_estimators=300, max_depth=6, learning_rate=0.1`
- **XGB-C5**: `n_estimators=500, max_depth=10, learning_rate=0.01`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| XGB-C1 | 0.2430 | 73.70% | 84.80% | Lớp 0 <br> Lớp 1 | 89.27% <br> 53.97% | 71.08% <br> 79.87% | 79.14% <br> 64.41% | 998 | 406 | 120 | 476 |
| XGB-C2 | 0.2205 | 72.00% | 83.98% | Lớp 0 <br> Lớp 1 | 89.59% <br> 51.93% | 68.02% <br> 81.38% | 77.33% <br> 63.40% | 955 | 449 | 111 | 485 |
| XGB-C3 | 0.2300 | 73.25% | 83.80% | Lớp 0 <br> Lớp 1 | 89.83% <br> 53.36% | 69.80% <br> 81.38% | 78.56% <br> 64.45% | 980 | 424 | 111 | 485 |
| XGB-C4 | 0.1560 | 68.95% | 81.19% | Lớp 0 <br> Lớp 1 | 89.67% <br> 48.77% | 63.03% <br> 82.89% | 74.03% <br> 61.40% | 885 | 519 | 102 | 494 |
| XGB-C5 | 0.1830 | 70.25% | 82.81% | Lớp 0 <br> Lớp 1 | 89.62% <br> 50.05% | 65.17% <br> 82.21% | 75.46% <br> 62.22% | 915 | 489 | 106 | 490 |

#### B. Không gian Full 11 Features (Có SMOTE = 0.5)

**Tham số chi tiết của các cấu hình:**
- **XGB-C1**: `n_estimators=100, max_depth=3, learning_rate=0.1`
- **XGB-C2**: `n_estimators=200, max_depth=5, learning_rate=0.05`
- **XGB-C3**: `n_estimators=150, max_depth=4, learning_rate=0.1`
- **XGB-C4**: `n_estimators=300, max_depth=6, learning_rate=0.1`
- **XGB-C5**: `n_estimators=500, max_depth=10, learning_rate=0.01`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| XGB-C1 | 0.2740 | 74.00% | 84.85% | Lớp 0 <br> Lớp 1 | 89.82% <br> 54.27% | 71.01% <br> 81.04% | 79.32% <br> 65.01% | 997 | 407 | 113 | 483 |
| XGB-C2 | 0.2525 | 71.85% | 83.52% | Lớp 0 <br> Lớp 1 | 89.34% <br> 51.77% | 68.02% <br> 80.87% | 77.23% <br> 63.13% | 955 | 449 | 114 | 482 |
| XGB-C3 | 0.2335 | 70.85% | 83.55% | Lớp 0 <br> Lớp 1 | 90.13% <br> 50.67% | 65.67% <br> 83.05% | 75.98% <br> 62.94% | 922 | 482 | 101 | 495 |
| XGB-C4 | 0.1845 | 69.05% | 81.36% | Lớp 0 <br> Lớp 1 | 89.21% <br> 48.85% | 63.60% <br> 81.88% | 74.26% <br> 61.19% | 893 | 511 | 108 | 488 |
| XGB-C5 | 0.1925 | 68.30% | 82.26% | Lớp 0 <br> Lớp 1 | 88.97% <br> 48.12% | 62.61% <br> 81.71% | 73.49% <br> 60.57% | 879 | 525 | 109 | 487 |

#### C. Không gian Full 11 Features (Có SMOTE = 0.6)

**Tham số chi tiết của các cấu hình:**
- **XGB-C1**: `n_estimators=100, max_depth=3, learning_rate=0.1`
- **XGB-C2**: `n_estimators=200, max_depth=5, learning_rate=0.05`
- **XGB-C3**: `n_estimators=150, max_depth=4, learning_rate=0.1`
- **XGB-C4**: `n_estimators=300, max_depth=6, learning_rate=0.1`
- **XGB-C5**: `n_estimators=500, max_depth=10, learning_rate=0.01`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| XGB-C1 | 0.2995 | 73.10% | 84.94% | Lớp 0 <br> Lớp 1 | 89.72% <br> 53.19% | 69.66% <br> 81.21% | 78.43% <br> 64.28% | 978 | 426 | 112 | 484 |
| XGB-C2 | 0.2660 | 70.75% | 83.78% | Lớp 0 <br> Lớp 1 | 90.19% <br> 50.56% | 65.46% <br> 83.22% | 75.86% <br> 62.90% | 919 | 485 | 100 | 496 |
| XGB-C3 | 0.2765 | 71.90% | 83.69% | Lớp 0 <br> Lớp 1 | 89.94% <br> 51.80% | 67.52% <br> 82.21% | 77.14% <br> 63.55% | 948 | 456 | 106 | 490 |
| XGB-C4 | 0.2075 | 68.50% | 81.01% | Lớp 0 <br> Lớp 1 | 89.73% <br> 48.34% | 62.25% <br> 83.22% | 73.51% <br> 61.16% | 874 | 530 | 100 | 496 |
| XGB-C5 | 0.2280 | 69.10% | 82.53% | Lớp 0 <br> Lớp 1 | 89.70% <br> 48.91% | 63.25% <br> 82.89% | 74.19% <br> 61.52% | 888 | 516 | 102 | 494 |

#### D. Không gian Full 11 Features (Có SMOTE = 0.7)

**Tham số chi tiết của các cấu hình:**
- **XGB-C1**: `n_estimators=100, max_depth=3, learning_rate=0.1`
- **XGB-C2**: `n_estimators=200, max_depth=5, learning_rate=0.05`
- **XGB-C3**: `n_estimators=150, max_depth=4, learning_rate=0.1`
- **XGB-C4**: `n_estimators=300, max_depth=6, learning_rate=0.1`
- **XGB-C5**: `n_estimators=500, max_depth=10, learning_rate=0.01`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| XGB-C1 | 0.3475 | 74.10% | 84.80% | Lớp 0 <br> Lớp 1 | 89.48% <br> 54.44% | 71.51% <br> 80.20% | 79.49% <br> 64.86% | 1004 | 400 | 118 | 478 |
| XGB-C2 | 0.3170 | 71.95% | 83.66% | Lớp 0 <br> Lớp 1 | 89.14% <br> 51.90% | 68.38% <br> 80.37% | 77.39% <br> 63.07% | 960 | 444 | 117 | 479 |
| XGB-C3 | 0.3270 | 73.15% | 84.08% | Lớp 0 <br> Lớp 1 | 89.59% <br> 53.26% | 69.87% <br> 80.87% | 78.51% <br> 64.22% | 981 | 423 | 114 | 482 |
| XGB-C4 | 0.2520 | 70.25% | 81.75% | Lớp 0 <br> Lớp 1 | 89.93% <br> 50.05% | 64.89% <br> 82.89% | 75.38% <br> 62.41% | 911 | 493 | 102 | 494 |
| XGB-C5 | 0.2415 | 68.30% | 82.43% | Lớp 0 <br> Lớp 1 | 89.77% <br> 48.16% | 61.89% <br> 83.39% | 73.27% <br> 61.06% | 869 | 535 | 99 | 497 |

#### E. Không gian MI Top 5 Features (Có SMOTE = 0.5) - Ngưỡng động tối ưu

**Tham số chi tiết của các cấu hình:**
- **XGB-C1**: `n_estimators=100, max_depth=3, learning_rate=0.1`
- **XGB-C2**: `n_estimators=200, max_depth=5, learning_rate=0.05`
- **XGB-C3**: `n_estimators=150, max_depth=4, learning_rate=0.1`
- **XGB-C4**: `n_estimators=300, max_depth=6, learning_rate=0.1`
- **XGB-C5**: `n_estimators=500, max_depth=10, learning_rate=0.01`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| XGB-C1 | 0.2610 | 73.65% | 84.11% | Lớp 0 <br> Lớp 1 | 89.76% <br> 53.85% | 70.51% <br> 81.04% | 78.98% <br> 64.70% | 990 | 414 | 113 | 483 |
| XGB-C2 | 0.2620 | 72.60% | 82.82% | Lớp 0 <br> Lớp 1 | 88.77% <br> 52.68% | 69.80% <br> 79.19% | 78.15% <br> 63.27% | 980 | 424 | 124 | 472 |
| XGB-C3 | 0.2510 | 71.80% | 83.17% | Lớp 0 <br> Lớp 1 | 88.67% <br> 51.75% | 68.59% <br> 79.36% | 77.35% <br> 62.65% | 963 | 441 | 123 | 473 |
| XGB-C4 | 0.2050 | 67.80% | 80.47% | Lớp 0 <br> Lớp 1 | 88.38% <br> 47.62% | 62.32% <br> 80.70% | 73.10% <br> 59.90% | 875 | 529 | 115 | 481 |
| XGB-C5 | 0.2035 | 67.25% | 80.48% | Lớp 0 <br> Lớp 1 | 89.05% <br> 47.17% | 60.83% <br> 82.38% | 72.28% <br> 59.99% | 854 | 550 | 105 | 491 |

#### F. Không gian MI Top 5 Features (Có SMOTE = 0.6) - Ngưỡng động tối ưu

**Tham số chi tiết của các cấu hình:**
- **XGB-C1**: `n_estimators=100, max_depth=3, learning_rate=0.1`
- **XGB-C2**: `n_estimators=200, max_depth=5, learning_rate=0.05`
- **XGB-C3**: `n_estimators=150, max_depth=4, learning_rate=0.1`
- **XGB-C4**: `n_estimators=300, max_depth=6, learning_rate=0.1`
- **XGB-C5**: `n_estimators=500, max_depth=10, learning_rate=0.01`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| XGB-C1 | 0.2930 | 73.30% | 83.98% | Lớp 0 <br> Lớp 1 | 89.62% <br> 53.44% | 70.09% <br> 80.87% | 78.66% <br> 64.35% | 984 | 420 | 114 | 482 |
| XGB-C2 | 0.2900 | 72.00% | 82.79% | Lớp 0 <br> Lớp 1 | 88.64% <br> 51.98% | 68.95% <br> 79.19% | 77.56% <br> 62.77% | 968 | 436 | 124 | 472 |
| XGB-C3 | 0.2735 | 70.85% | 82.91% | Lớp 0 <br> Lớp 1 | 88.91% <br> 50.69% | 66.81% <br> 80.37% | 76.29% <br> 62.17% | 938 | 466 | 117 | 479 |
| XGB-C4 | 0.2490 | 68.80% | 80.56% | Lớp 0 <br> Lớp 1 | 88.92% <br> 48.60% | 63.46% <br> 81.38% | 74.06% <br> 60.85% | 891 | 513 | 111 | 485 |
| XGB-C5 | 0.2560 | 68.00% | 80.60% | Lớp 0 <br> Lớp 1 | 87.82% <br> 47.78% | 63.18% <br> 79.36% | 73.49% <br> 59.65% | 887 | 517 | 123 | 473 |

#### G. Không gian MI Top 5 Features (Có SMOTE = 0.7) - Ngưỡng động tối ưu

**Tham số chi tiết của các cấu hình:**
- **XGB-C1**: `n_estimators=100, max_depth=3, learning_rate=0.1`
- **XGB-C2**: `n_estimators=200, max_depth=5, learning_rate=0.05`
- **XGB-C3**: `n_estimators=150, max_depth=4, learning_rate=0.1`
- **XGB-C4**: `n_estimators=300, max_depth=6, learning_rate=0.1`
- **XGB-C5**: `n_estimators=500, max_depth=10, learning_rate=0.01`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| XGB-C1 | 0.3260 | 72.75% | 83.97% | Lớp 0 <br> Lớp 1 | 89.73% <br> 52.77% | 69.09% <br> 81.38% | 78.07% <br> 64.03% | 970 | 434 | 111 | 485 |
| XGB-C2 | 0.3125 | 72.45% | 82.76% | Lớp 0 <br> Lớp 1 | 89.67% <br> 52.43% | 68.66% <br> 81.38% | 77.77% <br> 63.77% | 964 | 440 | 111 | 485 |
| XGB-C3 | 0.3130 | 71.45% | 82.99% | Lớp 0 <br> Lớp 1 | 88.96% <br> 51.34% | 67.74% <br> 80.20% | 76.91% <br> 62.61% | 951 | 453 | 118 | 478 |
| XGB-C4 | 0.2910 | 68.70% | 81.08% | Lớp 0 <br> Lớp 1 | 87.62% <br> 48.45% | 64.53% <br> 78.52% | 74.32% <br> 59.92% | 906 | 498 | 128 | 468 |
| XGB-C5 | 0.2950 | 68.60% | 80.30% | Lớp 0 <br> Lớp 1 | 88.19% <br> 48.37% | 63.82% <br> 79.87% | 74.05% <br> 60.25% | 896 | 508 | 120 | 476 |

### ❖ Thuật toán: Decision Tree

#### A. Không gian Full 11 Features (Không SMOTE)

**Tham số chi tiết của các cấu hình:**
- **DT-C1**: `max_depth=3, min_samples_split=2, class_weight='balanced'`
- **DT-C2**: `max_depth=5, min_samples_split=5, class_weight='balanced'`
- **DT-C3**: `max_depth=6, min_samples_leaf=20, class_weight='balanced'`
- **DT-C4**: `max_depth=8, min_samples_split=10, class_weight='balanced'`
- **DT-C5**: `max_depth=12, min_samples_split=20, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DT-C1 | 0.2940 | 63.60% | 82.31% | Lớp 0 <br> Lớp 1 | 94.13% <br> 44.65% | 51.35% <br> 92.45% | 66.45% <br> 60.22% | 721 | 683 | 45 | 551 |
| DT-C2 | 0.3755 | 72.10% | 83.15% | Lớp 0 <br> Lớp 1 | 89.46% <br> 52.05% | 68.30% <br> 81.04% | 77.46% <br> 63.39% | 959 | 445 | 113 | 483 |
| DT-C3 | 0.3860 | 71.00% | 83.74% | Lớp 0 <br> Lớp 1 | 89.92% <br> 50.83% | 66.10% <br> 82.55% | 76.19% <br> 62.92% | 928 | 476 | 104 | 492 |
| DT-C4 | 0.3340 | 69.55% | 79.79% | Lớp 0 <br> Lớp 1 | 88.11% <br> 49.32% | 65.46% <br> 79.19% | 75.11% <br> 60.79% | 919 | 485 | 124 | 472 |
| DT-C5 | 0.1160 | 63.25% | 76.99% | Lớp 0 <br> Lớp 1 | 88.14% <br> 43.81% | 55.06% <br> 82.55% | 67.78% <br> 57.24% | 773 | 631 | 104 | 492 |

#### B. Không gian Full 11 Features (Có SMOTE = 0.5)

**Tham số chi tiết của các cấu hình:**
- **DT-C1**: `max_depth=3, min_samples_split=2, class_weight='balanced'`
- **DT-C2**: `max_depth=5, min_samples_split=5, class_weight='balanced'`
- **DT-C3**: `max_depth=6, min_samples_leaf=20, class_weight='balanced'`
- **DT-C4**: `max_depth=8, min_samples_split=10, class_weight='balanced'`
- **DT-C5**: `max_depth=12, min_samples_split=20, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DT-C1 | 0.2880 | 63.60% | 82.03% | Lớp 0 <br> Lớp 1 | 94.13% <br> 44.65% | 51.35% <br> 92.45% | 66.45% <br> 60.22% | 721 | 683 | 45 | 551 |
| DT-C2 | 0.3575 | 71.05% | 82.85% | Lớp 0 <br> Lớp 1 | 89.32% <br> 50.89% | 66.74% <br> 81.21% | 76.40% <br> 62.57% | 937 | 467 | 112 | 484 |
| DT-C3 | 0.4105 | 70.80% | 83.51% | Lớp 0 <br> Lớp 1 | 89.81% <br> 50.62% | 65.88% <br> 82.38% | 76.01% <br> 62.71% | 925 | 479 | 105 | 491 |
| DT-C4 | 0.3270 | 70.85% | 79.85% | Lớp 0 <br> Lớp 1 | 88.98% <br> 50.69% | 66.74% <br> 80.54% | 76.27% <br> 62.22% | 937 | 467 | 116 | 480 |
| DT-C5 | 0.2500 | 66.55% | 76.69% | Lớp 0 <br> Lớp 1 | 87.54% <br> 46.43% | 61.04% <br> 79.53% | 71.93% <br> 58.63% | 857 | 547 | 122 | 474 |

#### C. Không gian Full 11 Features (Có SMOTE = 0.6)

**Tham số chi tiết của các cấu hình:**
- **DT-C1**: `max_depth=3, min_samples_split=2, class_weight='balanced'`
- **DT-C2**: `max_depth=5, min_samples_split=5, class_weight='balanced'`
- **DT-C3**: `max_depth=6, min_samples_leaf=20, class_weight='balanced'`
- **DT-C4**: `max_depth=8, min_samples_split=10, class_weight='balanced'`
- **DT-C5**: `max_depth=12, min_samples_split=20, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DT-C1 | 0.4485 | 63.60% | 82.04% | Lớp 0 <br> Lớp 1 | 94.13% <br> 44.65% | 51.35% <br> 92.45% | 66.45% <br> 60.22% | 721 | 683 | 45 | 551 |
| DT-C2 | 0.3685 | 72.15% | 83.03% | Lớp 0 <br> Lớp 1 | 89.69% <br> 52.09% | 68.16% <br> 81.54% | 77.46% <br> 63.57% | 957 | 447 | 110 | 486 |
| DT-C3 | 0.4090 | 71.40% | 83.43% | Lớp 0 <br> Lớp 1 | 89.62% <br> 51.26% | 67.02% <br> 81.71% | 76.69% <br> 63.00% | 941 | 463 | 109 | 487 |
| DT-C4 | 0.3375 | 68.50% | 79.63% | Lớp 0 <br> Lớp 1 | 87.94% <br> 48.27% | 63.89% <br> 79.36% | 74.01% <br> 60.03% | 897 | 507 | 123 | 473 |
| DT-C5 | 0.1945 | 63.30% | 76.91% | Lớp 0 <br> Lớp 1 | 87.31% <br> 43.74% | 55.84% <br> 80.87% | 68.11% <br> 56.77% | 784 | 620 | 114 | 482 |

#### D. Không gian Full 11 Features (Có SMOTE = 0.7)

**Tham số chi tiết của các cấu hình:**
- **DT-C1**: `max_depth=3, min_samples_split=2, class_weight='balanced'`
- **DT-C2**: `max_depth=5, min_samples_split=5, class_weight='balanced'`
- **DT-C3**: `max_depth=6, min_samples_leaf=20, class_weight='balanced'`
- **DT-C4**: `max_depth=8, min_samples_split=10, class_weight='balanced'`
- **DT-C5**: `max_depth=12, min_samples_split=20, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DT-C1 | 0.3015 | 63.10% | 81.92% | Lớp 0 <br> Lớp 1 | 94.40% <br> 44.32% | 50.43% <br> 92.95% | 65.74% <br> 60.02% | 708 | 696 | 42 | 554 |
| DT-C2 | 0.4010 | 72.20% | 83.43% | Lớp 0 <br> Lớp 1 | 90.08% <br> 52.12% | 67.88% <br> 82.38% | 77.42% <br> 63.85% | 953 | 451 | 105 | 491 |
| DT-C3 | 0.4325 | 72.25% | 83.15% | Lớp 0 <br> Lớp 1 | 89.20% <br> 52.24% | 68.80% <br> 80.37% | 77.68% <br> 63.32% | 966 | 438 | 117 | 479 |
| DT-C4 | 0.3360 | 69.75% | 79.35% | Lớp 0 <br> Lớp 1 | 88.52% <br> 49.53% | 65.38% <br> 80.03% | 75.22% <br> 61.19% | 918 | 486 | 119 | 477 |
| DT-C5 | 0.2345 | 66.20% | 76.88% | Lớp 0 <br> Lớp 1 | 89.06% <br> 46.25% | 59.12% <br> 82.89% | 71.06% <br> 59.38% | 830 | 574 | 102 | 494 |

#### E. Không gian MI Top 5 Features (Có SMOTE = 0.5) - Ngưỡng động tối ưu

**Tham số chi tiết của các cấu hình:**
- **DT-C1**: `max_depth=3, min_samples_split=2, class_weight='balanced'`
- **DT-C2**: `max_depth=5, min_samples_split=5, class_weight='balanced'`
- **DT-C3**: `max_depth=6, min_samples_leaf=20, class_weight='balanced'`
- **DT-C4**: `max_depth=8, min_samples_split=10, class_weight='balanced'`
- **DT-C5**: `max_depth=12, min_samples_split=20, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DT-C1 | 0.2880 | 63.60% | 82.03% | Lớp 0 <br> Lớp 1 | 94.13% <br> 44.65% | 51.35% <br> 92.45% | 66.45% <br> 60.22% | 721 | 683 | 45 | 551 |
| DT-C2 | 0.3335 | 70.60% | 83.18% | Lớp 0 <br> Lớp 1 | 89.46% <br> 50.41% | 65.88% <br> 81.71% | 75.88% <br> 62.36% | 925 | 479 | 109 | 487 |
| DT-C3 | 0.4315 | 73.15% | 82.68% | Lớp 0 <br> Lớp 1 | 89.30% <br> 53.29% | 70.16% <br> 80.20% | 78.58% <br> 64.03% | 985 | 419 | 118 | 478 |
| DT-C4 | 0.3925 | 72.00% | 79.22% | Lớp 0 <br> Lớp 1 | 88.09% <br> 52.02% | 69.52% <br> 77.85% | 77.71% <br> 62.37% | 976 | 428 | 132 | 464 |
| DT-C5 | 0.2860 | 64.70% | 73.12% | Lớp 0 <br> Lớp 1 | 85.11% <br> 44.53% | 60.26% <br> 75.17% | 70.56% <br> 55.93% | 846 | 558 | 148 | 448 |

#### F. Không gian MI Top 5 Features (Có SMOTE = 0.6) - Ngưỡng động tối ưu

**Tham số chi tiết của các cấu hình:**
- **DT-C1**: `max_depth=3, min_samples_split=2, class_weight='balanced'`
- **DT-C2**: `max_depth=5, min_samples_split=5, class_weight='balanced'`
- **DT-C3**: `max_depth=6, min_samples_leaf=20, class_weight='balanced'`
- **DT-C4**: `max_depth=8, min_samples_split=10, class_weight='balanced'`
- **DT-C5**: `max_depth=12, min_samples_split=20, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DT-C1 | 0.3035 | 63.60% | 82.06% | Lớp 0 <br> Lớp 1 | 94.13% <br> 44.65% | 51.35% <br> 92.45% | 66.45% <br> 60.22% | 721 | 683 | 45 | 551 |
| DT-C2 | 0.3610 | 71.00% | 82.77% | Lớp 0 <br> Lớp 1 | 89.02% <br> 50.85% | 66.95% <br> 80.54% | 76.42% <br> 62.34% | 940 | 464 | 116 | 480 |
| DT-C3 | 0.3425 | 69.05% | 82.51% | Lớp 0 <br> Lớp 1 | 89.53% <br> 48.86% | 63.32% <br> 82.55% | 74.18% <br> 61.38% | 889 | 515 | 104 | 492 |
| DT-C4 | 0.3225 | 67.40% | 79.82% | Lớp 0 <br> Lớp 1 | 88.45% <br> 47.26% | 61.61% <br> 81.04% | 72.63% <br> 59.70% | 865 | 539 | 113 | 483 |
| DT-C5 | 0.3130 | 62.85% | 76.18% | Lớp 0 <br> Lớp 1 | 87.77% <br> 43.47% | 54.70% <br> 82.05% | 67.40% <br> 56.83% | 768 | 636 | 107 | 489 |

#### G. Không gian MI Top 5 Features (Có SMOTE = 0.7) - Ngưỡng động tối ưu

**Tham số chi tiết của các cấu hình:**
- **DT-C1**: `max_depth=3, min_samples_split=2, class_weight='balanced'`
- **DT-C2**: `max_depth=5, min_samples_split=5, class_weight='balanced'`
- **DT-C3**: `max_depth=6, min_samples_leaf=20, class_weight='balanced'`
- **DT-C4**: `max_depth=8, min_samples_split=10, class_weight='balanced'`
- **DT-C5**: `max_depth=12, min_samples_split=20, class_weight='balanced'`

| Config | Ngưỡng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DT-C1 | 0.3015 | 63.60% | 82.28% | Lớp 0 <br> Lớp 1 | 94.13% <br> 44.65% | 51.35% <br> 92.45% | 66.45% <br> 60.22% | 721 | 683 | 45 | 551 |
| DT-C2 | 0.3465 | 72.35% | 83.00% | Lớp 0 <br> Lớp 1 | 89.22% <br> 52.35% | 68.95% <br> 80.37% | 77.78% <br> 63.40% | 968 | 436 | 117 | 479 |
| DT-C3 | 0.4495 | 73.00% | 82.79% | Lớp 0 <br> Lớp 1 | 88.50% <br> 53.19% | 70.73% <br> 78.36% | 78.62% <br> 63.36% | 993 | 411 | 129 | 467 |
| DT-C4 | 0.3285 | 69.10% | 78.99% | Lớp 0 <br> Lớp 1 | 89.22% <br> 48.90% | 63.68% <br> 81.88% | 74.31% <br> 61.23% | 894 | 510 | 108 | 488 |
| DT-C5 | 0.2225 | 62.60% | 75.12% | Lớp 0 <br> Lớp 1 | 87.79% <br> 43.29% | 54.27% <br> 82.21% | 67.08% <br> 56.71% | 762 | 642 | 106 | 490 |

