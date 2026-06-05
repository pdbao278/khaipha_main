# ĐỐI SÁNH HIỆU QUẢ CÁC MỐC ĐẶC TRƯNG MI (SMOTE = 0.5 - NGƯỠNG CỐ ĐỊNH)

- **Ngày thực hiện**: 2026-06-05
- **Tập dữ liệu**: `data_processed.csv`
- **Chiến lược chia**: Train 70%, Validation 10%, Test 20% (Random State = 42, Stratified)
- **Tỷ lệ SMOTE**: 0.5 (Cố định lớp thiểu số bằng 50% lớp đa số)
- **Ngưỡng quyết định**: Giữ cố định theo cấu hình tối ưu để đánh giá chính xác ảnh hưởng của việc giảm/tăng số lượng đặc trưng đầu vào từ 5 lên 11 đặc trưng.

---

## 📊 1. LOGISTIC REGRESSION (LR-C4)

**Tham số & Ngưỡng cố định**: `C=10.0, penalty='l2', class_weight='balanced'`, Ngưỡng = **`0.4455`**

| Số đặc trưng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5 (MI Top 5)** | 73.80% | 84.63% | Lớp 0 <br> Lớp 1 | 90.44% <br> **53.95%** | 70.09% <br> **82.55%** | 78.97% <br> **65.25%** | 984 | 420 | 104 | 492 |
| **6** | 73.35% | 84.75% | Lớp 0 <br> Lớp 1 | 90.09% <br> 53.47% | 69.94% <br> 81.38% | 78.75% <br> 64.54% | 982 | 422 | 111 | 485 |
| **7** | 73.45% | 84.73% | Lớp 0 <br> Lớp 1 | 90.26% <br> 53.58% | 69.94% <br> 81.71% | 78.81% <br> 64.72% | 982 | 422 | 109 | 487 |
| **8** | 73.65% | 84.79% | Lớp 0 <br> Lớp 1 | 89.93% <br> 53.85% | 70.58% <br> 80.87% | 79.10% <br> 64.65% | 991 | 413 | 114 | 482 |
| **9** | 73.70% | 85.05% | Lớp 0 <br> Lớp 1 | 89.76% <br> 53.91% | 70.58% <br> 81.04% | 79.06% <br> 64.75% | 991 | 413 | 113 | 483 |
| **10** | 73.75% | 85.06% | Lớp 0 <br> Lớp 1 | 89.77% <br> 53.97% | 70.66% <br> 81.04% | 79.10% <br> 64.79% | 992 | 412 | 113 | 483 |
| **11 (Full 11)** | **73.90%** | **85.07%** | Lớp 0 <br> Lớp 1 | 89.80% <br> 54.15% | 70.87% <br> 81.04% | 79.22% <br> 64.92% | 995 | 409 | 113 | 483 |

### Phân tích Logistic Regression:
- **Độ ổn định cực cao**: Độ lệch Accuracy giữa mốc 5 và mốc 11 chỉ là $0.1\%$, Recall biến động nhỏ trong khoảng $81.04\% - 82.55\%$.
- **Tối ưu ở mốc 5 đặc trưng**: Đạt F1-Score tốt nhất ($65.25\%$) và Recall cao nhất ($82.55\%$), cho thấy 5 đặc trưng MI chính là cốt lõi phân loại của mô hình LR.

---

## 📊 2. SVM (RBF) - SVM-C5

**Tham số & Ngưỡng cố định**: `C=100.0, kernel='rbf', gamma=0.01, class_weight='balanced'`, Ngưỡng = **`0.1765`**

| Số đặc trưng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5 (MI Top 5)** | **70.05%** | 83.98% | Lớp 0 <br> Lớp 1 | 91.45% <br> **49.85%** | 63.25% <br> **86.07%** | 74.78% <br> **63.14%** | 888 | 516 | 83 | 513 |
| **6** | **71.00%** | 84.40% | Lớp 0 <br> Lớp 1 | 90.95% <br> 50.80% | 65.17% <br> 84.73% | 75.93% <br> 63.52% | 915 | 489 | 91 | 505 |
| **7** | 70.70% | 84.23% | Lớp 0 <br> Lớp 1 | 91.31% <br> 50.50% | 64.39% <br> 85.57% | 75.49% <br> 63.51% | 904 | 500 | 86 | 510 |
| **8** | 70.30% | 84.29% | Lớp 0 <br> Lớp 1 | 90.99% <br> 50.10% | 64.03% <br> 85.07% | 75.14% <br> 63.06% | 899 | 505 | 89 | 507 |
| **9** | 70.10% | **84.68%** | Lớp 0 <br> Lớp 1 | 91.29% <br> 49.90% | 63.46% <br> 85.74% | 74.84% <br> 63.09% | 891 | 513 | 85 | 511 |
| **10** | 70.30% | 84.66% | Lớp 0 <br> Lớp 1 | 91.16% <br> 50.10% | 63.89% <br> 85.40% | 75.09% <br> 63.15% | 897 | 507 | 87 | 509 |
| **11 (Full 11)** | 70.05% | 84.64% | Lớp 0 <br> Lớp 1 | 91.20% <br> 49.85% | 63.46% <br> 85.57% | 74.84% <br> 63.00% | 891 | 513 | 86 | 510 |

### Phân tích SVM:
- **Khả năng duy trì độ nhạy**: SVM duy trì Recall cực cao ($\ge 84.73\%$) ở tất cả các mốc đặc trưng dưới ngưỡng cố định `0.1765`.
- **Tối ưu**: Mốc 5 đặc trưng cho Recall tốt nhất ($86.07\%$), giúp hạn chế tối đa việc bỏ sót ca bệnh (chỉ lọt 83 ca FN).

---

## 📊 3. RANDOM FOREST (RF-C3)

**Tham số & Ngưỡng cố định**: `max_depth=8, n_estimators=200, class_weight='balanced'`, Ngưỡng = **`0.3875`**

| Số đặc trưng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5 (MI Top 5)** | **72.65%** | 82.77% | Lớp 0 <br> Lớp 1 | 88.71% <br> **52.74%** | 69.94% <br> **79.03%** | 78.22% <br> **63.26%** | 982 | 422 | 125 | 471 |
| **6** | 71.45% | 83.58% | Lớp 0 <br> Lớp 1 | 89.40% <br> 51.33% | 67.31% <br> 81.21% | 76.80% <br> 62.90% | 945 | 459 | 112 | 484 |
| **7** | 71.90% | 83.74% | Lớp 0 <br> Lớp 1 | 89.79% <br> 51.80% | 67.66% <br> 81.88% | 77.20% <br> 63.46% | 950 | 454 | 108 | 488 |
| **8** | 71.75% | 84.09% | Lớp 0 <br> Lớp 1 | 90.38% <br> 51.61% | 66.88% <br> 83.22% | 76.84% <br> 63.71% | 939 | 465 | 100 | 496 |
| **9** | 72.30% | 84.36% | Lớp 0 <br> Lớp 1 | 90.25% <br> 52.22% | 67.88% <br> 82.72% | 77.51% <br> 64.03% | 953 | 451 | 103 | 493 |
| **10** | 71.55% | 84.39% | Lớp 0 <br> Lớp 1 | 90.34% <br> 51.40% | 66.60% <br> 83.22% | 76.67% <br> 63.55% | 935 | 469 | 100 | 496 |
| **11 (Full 11)** | 71.60% | **84.58%** | Lớp 0 <br> Lớp 1 | 90.43% <br> 51.45% | 66.50% <br> 83.39% | 76.64% <br> 63.64% | 935 | 469 | 99 | 497 |

### Phân tích Random Forest:
- **Hiệu quả tỷ lệ thuận với số lượng đặc trưng**: Recall tăng rõ rệt từ $79.03\%$ (5 đặc trưng) lên $83.39\%$ (11 đặc trưng). F1-Score cũng tăng dần và đạt đỉnh tại 11 đặc trưng ($63.64\%$).
- **Kết luận**: RF tận dụng tốt sự kết hợp thông tin phi tuyến từ nhiều đặc trưng.

---

## 📊 4. XGBOOST (XGB-C3)

**Tham số & Ngưỡng cố định**: `max_depth=4, learning_rate=0.1, n_estimators=150`, Ngưỡng = **`0.2335`**

| Số đặc trưng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5 (MI Top 5)** | 70.05% | 83.17% | Lớp 0 <br> Lớp 1 | 89.73% <br> **49.85%** | 64.74% <br> **82.55%** | 75.22% <br> **62.16%** | 909 | 495 | 104 | 492 |
| **6** | 69.55% | 83.02% | Lớp 0 <br> Lớp 1 | 89.63% <br> 49.35% | 64.03% <br> 82.55% | 74.70% <br> 61.77% | 899 | 505 | 104 | 492 |
| **7** | 70.30% | 83.14% | Lớp 0 <br> Lớp 1 | 90.10% <br> 50.10% | 64.81% <br> 83.22% | 75.39% <br> 62.55% | 910 | 494 | 100 | 496 |
| **8** | 71.05% | 83.18% | Lớp 0 <br> Lớp 1 | 90.24% <br> 50.87% | 65.88% <br> 83.22% | 76.16% <br> 63.14% | 925 | 479 | 100 | 496 |
| **9** | 70.95% | **83.70%** | Lớp 0 <br> Lớp 1 | 90.15% <br> 50.77% | 65.81% <br> 83.05% | 76.08% <br> 63.02% | 924 | 480 | 101 | 495 |
| **10** | **71.45%** | 83.70% | Lớp 0 <br> Lớp 1 | 90.32% <br> 51.29% | 66.45% <br> 83.22% | 76.57% <br> 63.47% | 933 | 471 | 100 | 496 |
| **11 (Full 11)** | 70.85% | 83.55% | Lớp 0 <br> Lớp 1 | 90.13% <br> 50.67% | 65.67% <br> 83.05% | 75.98% <br> 62.94% | 922 | 482 | 101 | 495 |

### Phân tích XGBoost:
- XGBoost duy trì Recall ổn định trên $82.5\%$. Khi bổ sung thêm đặc trưng lên mốc 8 và 10, mô hình đạt F1-Score tối ưu hơn ($63.14\%$ và $63.47\%$) nhờ tăng Precision đầu ra.

---

## 📊 5. DECISION TREE (DT-C3)

**Tham số & Ngưỡng cố định**: `max_depth=6, min_samples_leaf=20, class_weight='balanced'`, Ngưỡng = **`0.4105`**

| Số đặc trưng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5 (MI Top 5)** | **71.65%** | 82.68% | Lớp 0 <br> Lớp 1 | 89.44% <br> **51.54%** | 67.59% <br> **81.21%** | 77.00% <br> **63.06%** | 949 | 455 | 112 | 484 |
| **6** | 71.75% | 83.32% | Lớp 0 <br> Lớp 1 | 88.74% <br> 51.69% | 68.45% <br> 79.53% | 77.30% <br> 62.66% | 961 | 443 | 122 | 474 |
| **7** | 71.75% | 83.25% | Lớp 0 <br> Lớp 1 | 88.66% <br> 51.69% | 68.52% <br> 79.36% | 77.34% <br> 62.61% | 962 | 442 | 123 | 473 |
| **8** | 71.30% | 83.28% | Lớp 0 <br> Lớp 1 | 89.60% <br> 51.16% | 66.88% <br> 81.71% | 76.62% <br> 62.92% | 939 | 465 | 109 | 487 |
| **9** | 70.80% | **83.54%** | Lớp 0 <br> Lớp 1 | 89.88% <br> 50.62% | 65.81% <br> 82.55% | 75.99% <br> 62.76% | 924 | 480 | 104 | 492 |
| **10** | 70.80% | 83.51% | Lớp 0 <br> Lớp 1 | 89.81% <br> 50.62% | 65.88% <br> 82.38% | 76.01% <br> 62.71% | 925 | 479 | 105 | 491 |
| **11 (Full 11)** | 70.80% | 83.51% | Lớp 0 <br> Lớp 1 | 89.81% <br> 50.62% | 65.88% <br> 82.38% | 76.01% <br> 62.71% | 925 | 479 | 105 | 491 |

### Phân tích Decision Tree:
- DT-C3 đạt độ ổn định ở các mốc từ 9 đến 11 đặc trưng (chỉ số trùng khớp hoàn toàn do cấu trúc phân nhánh cây quyết định không đổi khi bổ sung các thuộc tính có điểm MI rất thấp). Mốc 5 đặc trưng cho sự cân bằng F1-Score tốt nhất ($63.06\%$).

---

## 📊 6. ĐỐI SÁNH CÁC THUẬT TOÁN GOM THEO TỪNG MỐC SỐ ĐẶC TRƯNG

Dưới đây là bảng đối sánh trực tiếp giữa các thuật toán khi được đặt cùng một số lượng đặc trưng đầu vào (ở tỷ lệ SMOTE = 0.5 cố định).

### ❖ Mốc 5 đặc trưng (MI Top 5)

| Thuật toán | Mã cấu hình | Ngưỡng cắt | Accuracy | AUC-ROC | Precision (Lớp 1) | Recall (Lớp 1) | F1-Score (Lớp 1) | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | LR-C4 | `0.4455` | **73.80%** | **84.63%** | **53.95%** | 82.55% | **65.25%** | 984 | 420 | 104 | 492 |
| **SVM (RBF)** | SVM-C5 | `0.1765` | 70.05% | 83.98% | 49.85% | **86.07%** | 63.14% | 888 | 516 | 83 | 513 |
| **Random Forest** | RF-C3 | `0.3875` | 72.65% | 82.77% | 52.74% | 79.03% | 63.26% | 982 | 422 | 125 | 471 |
| **XGBoost** | XGB-C3 | `0.2335` | 70.05% | 83.17% | 49.85% | 82.55% | 62.16% | 909 | 495 | 104 | 492 |
| **Decision Tree** | DT-C3 | `0.4105` | 71.65% | 82.68% | 51.54% | 81.21% | 63.06% | 949 | 455 | 112 | 484 |

### ❖ Mốc 6 đặc trưng

| Thuật toán | Mã cấu hình | Ngưỡng cắt | Accuracy | AUC-ROC | Precision (Lớp 1) | Recall (Lớp 1) | F1-Score (Lớp 1) | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | LR-C4 | `0.4455` | **73.35%** | **84.75%** | **53.47%** | 81.38% | **64.54%** | 982 | 422 | 111 | 485 |
| **SVM (RBF)** | SVM-C5 | `0.1765` | 71.00% | 84.40% | 50.80% | **84.73%** | 63.52% | 915 | 489 | 91 | 505 |
| **Random Forest** | RF-C3 | `0.3875` | 71.45% | 83.58% | 51.33% | 81.21% | 62.90% | 945 | 459 | 112 | 484 |
| **XGBoost** | XGB-C3 | `0.2335` | 69.55% | 83.02% | 49.35% | 82.55% | 61.77% | 899 | 505 | 104 | 492 |
| **Decision Tree** | DT-C3 | `0.4105` | 71.75% | 83.32% | 51.69% | 79.53% | 62.66% | 961 | 443 | 122 | 474 |

### ❖ Mốc 7 đặc trưng

| Thuật toán | Mã cấu hình | Ngưỡng cắt | Accuracy | AUC-ROC | Precision (Lớp 1) | Recall (Lớp 1) | F1-Score (Lớp 1) | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | LR-C4 | `0.4455` | **73.45%** | **84.73%** | **53.58%** | 81.71% | **64.72%** | 982 | 422 | 109 | 487 |
| **SVM (RBF)** | SVM-C5 | `0.1765` | 70.70% | 84.23% | 50.50% | **85.57%** | 63.51% | 904 | 500 | 86 | 510 |
| **Random Forest** | RF-C3 | `0.3875` | 71.90% | 83.74% | 51.80% | 81.88% | 63.46% | 950 | 454 | 108 | 488 |
| **XGBoost** | XGB-C3 | `0.2335` | 70.30% | 83.14% | 50.10% | 83.22% | 62.55% | 910 | 494 | 100 | 496 |
| **Decision Tree** | DT-C3 | `0.4105` | 71.75% | 83.25% | 51.69% | 79.36% | 62.61% | 962 | 442 | 123 | 473 |

### ❖ Mốc 8 đặc trưng

| Thuật toán | Mã cấu hình | Ngưỡng cắt | Accuracy | AUC-ROC | Precision (Lớp 1) | Recall (Lớp 1) | F1-Score (Lớp 1) | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | LR-C4 | `0.4455` | **73.65%** | **84.79%** | **53.85%** | 80.87% | **64.65%** | 991 | 413 | 114 | 482 |
| **SVM (RBF)** | SVM-C5 | `0.1765` | 70.30% | 84.29% | 50.10% | **85.07%** | 63.06% | 899 | 505 | 89 | 507 |
| **Random Forest** | RF-C3 | `0.3875` | 71.75% | 84.09% | 51.61% | 83.22% | 63.71% | 939 | 465 | 100 | 496 |
| **XGBoost** | XGB-C3 | `0.2335` | 71.05% | 83.18% | 50.87% | 83.22% | 63.14% | 925 | 479 | 100 | 496 |
| **Decision Tree** | DT-C3 | `0.4105` | 71.30% | 83.28% | 51.16% | 81.71% | 62.92% | 939 | 465 | 109 | 487 |

### ❖ Mốc 9 đặc trưng

| Thuật toán | Mã cấu hình | Ngưỡng cắt | Accuracy | AUC-ROC | Precision (Lớp 1) | Recall (Lớp 1) | F1-Score (Lớp 1) | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | LR-C4 | `0.4455` | **73.70%** | **85.05%** | **53.91%** | 81.04% | **64.75%** | 991 | 413 | 113 | 483 |
| **SVM (RBF)** | SVM-C5 | `0.1765` | 70.10% | 84.68% | 49.90% | **85.74%** | 63.09% | 891 | 513 | 85 | 511 |
| **Random Forest** | RF-C3 | `0.3875` | 72.30% | 84.36% | 52.22% | 82.72% | 64.03% | 953 | 451 | 103 | 493 |
| **XGBoost** | XGB-C3 | `0.2335` | 70.95% | 83.70% | 50.77% | 83.05% | 63.02% | 924 | 480 | 101 | 495 |
| **Decision Tree** | DT-C3 | `0.4105` | 70.80% | 83.54% | 50.62% | 82.55% | 62.76% | 924 | 480 | 104 | 492 |

### ❖ Mốc 10 đặc trưng

| Thuật toán | Mã cấu hình | Ngưỡng cắt | Accuracy | AUC-ROC | Precision (Lớp 1) | Recall (Lớp 1) | F1-Score (Lớp 1) | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | LR-C4 | `0.4455` | **73.75%** | **85.06%** | **53.97%** | 81.04% | **64.79%** | 992 | 412 | 113 | 483 |
| **SVM (RBF)** | SVM-C5 | `0.1765` | 70.30% | 84.66% | 50.10% | **85.40%** | 63.15% | 897 | 507 | 87 | 509 |
| **Random Forest** | RF-C3 | `0.3875` | 71.55% | 84.39% | 51.40% | 83.22% | 63.55% | 935 | 469 | 100 | 496 |
| **XGBoost** | XGB-C3 | `0.2335` | 71.45% | 83.70% | 51.29% | 83.22% | 63.47% | 933 | 471 | 100 | 496 |
| **Decision Tree** | DT-C3 | `0.4105` | 70.80% | 83.51% | 50.62% | 82.38% | 62.71% | 925 | 479 | 105 | 491 |

### ❖ Mốc 11 đặc trưng (Full 11)

| Thuật toán | Mã cấu hình | Ngưỡng cắt | Accuracy | AUC-ROC | Precision (Lớp 1) | Recall (Lớp 1) | F1-Score (Lớp 1) | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | LR-C4 | `0.4455` | **73.90%** | **85.07%** | **54.15%** | 81.04% | **64.92%** | 995 | 409 | 113 | 483 |
| **SVM (RBF)** | SVM-C5 | `0.1765` | 70.05% | 84.64% | 49.85% | **85.57%** | 63.00% | 891 | 513 | 86 | 510 |
| **Random Forest** | RF-C3 | `0.3875` | 71.60% | 84.58% | 51.45% | 83.39% | 63.64% | 935 | 469 | 99 | 497 |
| **XGBoost** | XGB-C3 | `0.2335` | 70.85% | 83.55% | 50.67% | 83.05% | 62.94% | 922 | 482 | 101 | 495 |
| **Decision Tree** | DT-C3 | `0.4105` | 70.80% | 83.51% | 50.62% | 82.38% | 62.71% | 925 | 479 | 105 | 491 |

---

## 🎯 TỔNG KẾT VÀ KHUYẾN NGHỊ RÚT RA

### 📈 Bảng Tổng Hợp So Sánh Sự Biến Đổi (MI=5 vs Full 11)

| Thuật toán | MI = 5<br>Accuracy | Full 11<br>Accuracy | MI = 5<br>Recall (Lớp 1) | Full 11<br>Recall (Lớp 1) | MI = 5<br>F1-Score | Full 11<br>F1-Score | Xu hướng biến đổi khi tăng đặc trưng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 73.80% | **73.90%** | **82.55%** | 81.04% | **65.25%** | 64.92% | ⚖️ Cực kỳ ổn định, mốc 5 đặc trưng tốt nhất |
| **SVM (RBF)** | 70.05% | 70.05% | **86.07%** | 85.57% | **63.14%** | 63.00% | 🔬 Recall cực nhạy, mốc 5 đặc trưng tối ưu |
| **Random Forest** | **72.65%** | 71.60% | 79.03% | **83.39%** | 63.26% | **63.64%** | 📈 Tăng mạnh Recall và F1 khi thêm đặc trưng |
| **XGBoost** | 70.05% | **70.85%** | 82.55% | **83.05%** | 62.16% | **62.94%** | 📈 Tăng nhẹ hiệu năng tổng thể |
| **Decision Tree** | **71.65%** | 70.80% | 81.21% | **82.38%** | **63.06%** | 62.71% | ⚖️ Ổn định sau mốc 9 đặc trưng |

### 🎖️ Khuyến Nghị Chọn Mốc Đặc Trưng Để Ưu Tiên Recall (Độ Nhạy Phát Hiện Bệnh):

Nếu mục tiêu hàng đầu là **tối đa hóa Recall** để tránh bỏ sót các ca đột quỵ (giảm thiểu False Negatives), đây là chiến lược chọn mốc đặc trưng tối ưu cho từng trường hợp:

#### 1️⃣ **Lựa chọn Tối ưu Tuyệt đối (Absolute Best): Chọn mốc 5 đặc trưng kết hợp SVM (SVM-C5)**
* **Hiệu năng**: Đạt Recall **86.07%** (cao nhất trong toàn bộ thực nghiệm). Chỉ bỏ sót **83 ca bệnh** (False Negative = 83) trong tập Test.
* **Đặc trưng cần dùng (5)**: `['Hypertension', 'Age', 'Heart_Disease', 'Avg_Glucose', 'Diabetes']`.
* **Ưu điểm**: Cực kỳ tiết kiệm chi phí thu thập thông tin lâm sàng (chỉ cần 5 thuộc tính cơ bản dễ đo đạc) nhưng lại cho độ nhạy phát hiện bệnh cao nhất.

#### 2️⃣ **Lựa chọn Cân bằng tốt nhất giữa Recall & Accuracy: Chọn mốc 5 đặc trưng kết hợp Logistic Regression (LR-C4)**
* **Hiệu năng**: Đạt Recall **82.55%**, đi kèm Accuracy **73.80%** (vượt trội so với mức $70.05\%$ của SVM).
* **Lợi ích**: Giảm thiểu được gần **100 ca cảnh báo giả** (False Positive = 420 so với 516 của SVM) trong khi vẫn duy trì được độ nhạy sàng lọc cao trên $82\%$.

#### 3️⃣ **Lưu ý đặc biệt khi sử dụng các mô hình Ensemble (Random Forest, XGBoost): Bắt buộc dùng mốc 11 đặc trưng**
* **Lý do**: Các mô hình dạng cây tập hợp (Ensemble) bị sụt giảm Recall rất nhiều khi giảm số đặc trưng (ví dụ: Random Forest giảm từ **83.39%** ở mốc 11 đặc trưng xuống chỉ còn **79.03%** ở mốc 5 đặc trưng).
* **Khuyến nghị**: Nếu dự án bắt buộc sử dụng Random Forest hoặc XGBoost vì lý do kỹ thuật khác, **phải sử dụng mốc 11 đặc trưng** để đảm bảo Recall đạt mục tiêu sàng lọc ($\ge 83\%$).

---

**Báo cáo cập nhật ngày**: 2026-06-05  
**Nguồn số liệu thực nghiệm**: `run_fixed_threshold_comparison.py` (SMOTE = 0.5)
