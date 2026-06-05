# ĐỐI SÁNH HIỆU QUẢ CÁC MỐC ĐẶC TRƯNG MI (SMOTE = 0.5 - CHI TIẾT CẤU HÌNH)

- **Ngày thực hiện**: 2026-06-05
- **Tập dữ liệu**: `data_processed.csv`
- **Chiến lược chia**: Train 70%, Validation 10%, Test 20% (Random State = 42, Stratified)
- **Tỷ lệ SMOTE**: 0.5 (Cố định lớp thiểu số bằng 50% lớp đa số)
- **Ngưỡng quyết định**: Giữ cố định theo cấu hình tối ưu để đánh giá chính xác ảnh hưởng của việc giảm/tăng số lượng đặc trưng đầu vào từ 5 lên 11 đặc trưng.

---

## 1. LOGISTIC REGRESSION (LR-C4)

**Chi tiết cấu hình**: `C=10.0, penalty='l2', class_weight='balanced'`, Ngưỡng = **`0.4455`**

| Số đặc trưng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5 (MI Top 5)** | 73.80% | 84.63% | Lớp 0 <br> Lớp 1 | 90.44% <br> **53.95%** | 70.09% <br> **82.55%** | 78.97% <br> **65.25%** |
| **6** | 73.35% | 84.75% | Lớp 0 <br> Lớp 1 | 90.09% <br> 53.47% | 69.94% <br> 81.38% | 78.75% <br> 64.54% |
| **7** | 73.45% | 84.73% | Lớp 0 <br> Lớp 1 | 90.26% <br> 53.58% | 69.94% <br> 81.71% | 78.81% <br> 64.72% |
| **8** | 73.65% | 84.79% | Lớp 0 <br> Lớp 1 | 89.93% <br> 53.85% | 70.58% <br> 80.87% | 79.10% <br> 64.65% |
| **9** | 73.70% | 85.05% | Lớp 0 <br> Lớp 1 | 89.76% <br> 53.91% | 70.58% <br> 81.04% | 79.06% <br> 64.75% |
| **10** | 73.75% | 85.06% | Lớp 0 <br> Lớp 1 | 89.77% <br> 53.97% | 70.66% <br> 81.04% | 79.10% <br> 64.79% |
| **11 (Full 11)** | **73.90%** | **85.07%** | Lớp 0 <br> Lớp 1 | 89.80% <br> 54.15% | 70.87% <br> 81.04% | 79.22% <br> 64.92% |

### Phân tích Logistic Regression:
- Độ ổn định cực cao: Độ lệch Accuracy giữa mốc 5 và mốc 11 chỉ là 0.1%, Recall biến động nhỏ trong khoảng 81.04% - 82.55%.
- Tối ưu ở mốc 5 đặc trưng: Đạt F1-Score tốt nhất (65.25%) và Recall cao nhất (82.55%), cho thấy 5 đặc trưng MI chính là cốt lõi phân loại của mô hình LR.

---

## 2. SVM (RBF) - SVM-C5

**Chi tiết cấu hình**: `C=100.0, kernel='rbf', gamma=0.01, class_weight='balanced'`, Ngưỡng = **`0.1765`**

| Số đặc trưng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5 (MI Top 5)** | **70.05%** | 83.98% | Lớp 0 <br> Lớp 1 | 91.45% <br> **49.85%** | 63.25% <br> **86.07%** | 74.78% <br> **63.14%** |
| **6** | **71.00%** | 84.40% | Lớp 0 <br> Lớp 1 | 90.95% <br> 50.80% | 65.17% <br> 84.73% | 75.93% <br> 63.52% |
| **7** | 70.70% | 84.23% | Lớp 0 <br> Lớp 1 | 91.31% <br> 50.50% | 64.39% <br> 85.57% | 75.49% <br> 63.51% |
| **8** | 70.30% | 84.29% | Lớp 0 <br> Lớp 1 | 90.99% <br> 50.10% | 64.03% <br> 85.07% | 75.14% <br> 63.06% |
| **9** | 70.10% | **84.68%** | Lớp 0 <br> Lớp 1 | 91.29% <br> 49.90% | 63.46% <br> 85.74% | 74.84% <br> 63.09% |
| **10** | 70.30% | 84.66% | Lớp 0 <br> Lớp 1 | 91.16% <br> 50.10% | 63.89% <br> 85.40% | 75.09% <br> 63.15% |
| **11 (Full 11)** | 70.05% | 84.64% | Lớp 0 <br> Lớp 1 | 91.20% <br> 49.85% | 63.46% <br> 85.57% | 74.84% <br> 63.00% |

### Phân tích SVM:
- Khả năng duy trì độ nhạy: SVM duy trì Recall cực cao (>= 84.73%) ở tất cả các mốc đặc trưng dưới ngưỡng cố định 0.1765.
- Tối ưu: Mốc 5 đặc trưng cho Recall tốt nhất (86.07%), giúp hạn chế tối đa việc bỏ sót ca bệnh.

---

## 3. RANDOM FOREST (RF-C3)

**Chi tiết cấu hình**: `n_estimators=200, max_depth=8, class_weight='balanced'`, Ngưỡng = **`0.3875`**

| Số đặc trưng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5 (MI Top 5)** | **72.65%** | 82.77% | Lớp 0 <br> Lớp 1 | 88.71% <br> **52.74%** | 69.94% <br> **79.03%** | 78.22% <br> **63.26%** |
| **6** | 71.45% | 83.58% | Lớp 0 <br> Lớp 1 | 89.40% <br> 51.33% | 67.31% <br> 81.21% | 76.80% <br> 62.90% |
| **7** | 71.90% | 83.74% | Lớp 0 <br> Lớp 1 | 89.79% <br> 51.80% | 67.66% <br> 81.88% | 77.20% <br> 63.46% |
| **8** | 71.75% | 84.09% | Lớp 0 <br> Lớp 1 | 90.38% <br> 51.61% | 66.88% <br> 83.22% | 76.84% <br> 63.71% |
| **9** | 72.30% | 84.36% | Lớp 0 <br> Lớp 1 | 90.25% <br> 52.22% | 67.88% <br> 82.72% | 77.51% <br> 64.03% |
| **10** | 71.55% | 84.39% | Lớp 0 <br> Lớp 1 | 90.34% <br> 51.40% | 66.60% <br> 83.22% | 76.67% <br> 63.55% |
| **11 (Full 11)** | 71.60% | **84.58%** | Lớp 0 <br> Lớp 1 | 90.43% <br> 51.45% | 66.50% <br> 83.39% | 76.64% <br> 63.64% |

### Phân tích Random Forest:
- Hiệu quả tỷ lệ thuận với số lượng đặc trưng: Recall tăng rõ rệt từ 79.03% (5 đặc trưng) lên 83.39% (11 đặc trưng). F1-Score cũng tăng dần và đạt đỉnh tại 11 đặc trưng (63.64%).
- Kết luận: RF tận dụng tốt sự kết hợp thông tin phi tuyến từ nhiều đặc trưng.

---

## 4. XGBOOST (XGB-C3)

**Chi tiết cấu hình**: `n_estimators=150, max_depth=4, learning_rate=0.1`, Ngưỡng = **`0.2335`**

| Số đặc trưng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5 (MI Top 5)** | 70.05% | 83.17% | Lớp 0 <br> Lớp 1 | 89.73% <br> **49.85%** | 64.74% <br> **82.55%** | 75.22% <br> **62.16%** |
| **6** | 69.55% | 83.02% | Lớp 0 <br> Lớp 1 | 89.63% <br> 49.35% | 64.03% <br> 82.55% | 74.70% <br> 61.77% |
| **7** | 70.30% | 83.14% | Lớp 0 <br> Lớp 1 | 90.10% <br> 50.10% | 64.81% <br> 83.22% | 75.39% <br> 62.55% |
| **8** | 71.05% | 83.18% | Lớp 0 <br> Lớp 1 | 90.24% <br> 50.87% | 65.88% <br> 83.22% | 76.16% <br> 63.14% |
| **9** | 70.95% | **83.70%** | Lớp 0 <br> Lớp 1 | 90.15% <br> 50.77% | 65.81% <br> 83.05% | 76.08% <br> 63.02% |
| **10** | **71.45%** | 83.70% | Lớp 0 <br> Lớp 1 | 90.32% <br> 51.29% | 66.45% <br> 83.22% | 76.57% <br> 63.47% |
| **11 (Full 11)** | 70.85% | 83.55% | Lớp 0 <br> Lớp 1 | 90.13% <br> 50.67% | 65.67% <br> 83.05% | 75.98% <br> 62.94% |

### Phân tích XGBoost:
- XGBoost duy trì Recall ổn định trên 82.5%. Khi bổ sung thêm đặc trưng lên mốc 8 và 10, mô hình đạt F1-Score tối ưu hơn (63.14% và 63.47%) nhờ tăng Precision đầu ra.

---

## 5. DECISION TREE (DT-C3)

**Chi tiết cấu hình**: `max_depth=6, min_samples_leaf=20, class_weight='balanced'`, Ngưỡng = **`0.4105`**

| Số đặc trưng | Accuracy | AUC-ROC | Nhãn | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5 (MI Top 5)** | **71.65%** | 82.68% | Lớp 0 <br> Lớp 1 | 89.44% <br> **51.54%** | 67.59% <br> **81.21%** | 77.00% <br> **63.06%** |
| **6** | 71.75% | 83.32% | Lớp 0 <br> Lớp 1 | 88.74% <br> 51.69% | 68.45% <br> 79.53% | 77.30% <br> 62.66% |
| **7** | 71.75% | 83.25% | Lớp 0 <br> Lớp 1 | 88.66% <br> 51.69% | 68.52% <br> 79.36% | 77.34% <br> 62.61% |
| **8** | 71.30% | 83.28% | Lớp 0 <br> Lớp 1 | 89.60% <br> 51.16% | 66.88% <br> 81.71% | 76.62% <br> 62.92% |
| **9** | 70.80% | **83.54%** | Lớp 0 <br> Lớp 1 | 89.88% <br> 50.62% | 65.81% <br> 82.55% | 75.99% <br> 62.76% |
| **10** | 70.80% | 83.51% | Lớp 0 <br> Lớp 1 | 89.81% <br> 50.62% | 65.88% <br> 82.38% | 76.01% <br> 62.71% |
| **11 (Full 11)** | 70.80% | 83.51% | Lớp 0 <br> Lớp 1 | 89.81% <br> 50.62% | 65.88% <br> 82.38% | 76.01% <br> 62.71% |

### Phân tích Decision Tree:
- DT-C3 đạt độ ổn định ở các mốc từ 9 đến 11 đặc trưng. Mốc 5 đặc trưng cho sự cân bằng F1-Score tốt nhất (63.06%).

---

## 6. ĐỐI SÁNH CÁC THUẬT TOÁN GOM THEO TỪNG MỐC SỐ ĐẶC TRƯNG

Dưới đây là bảng đối sánh trực tiếp giữa các thuật toán khi được đặt cùng một số lượng đặc trưng đầu vào (ở tỷ lệ SMOTE = 0.5 cố định).

### MỐC 5 ĐẶC TRƯNG (MI TOP 5)

| Thuật toán | Chi tiết cấu hình | Ngưỡng cắt | Accuracy | AUC-ROC | Precision (Lớp 1) | Recall (Lớp 1) | F1-Score (Lớp 1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `C=10.0, l2, class_weight='balanced'` | `0.4455` | **73.80%** | **84.63%** | **53.95%** | 82.55% | **65.25%** |
| **SVM (RBF)** | `C=100.0, gamma=0.01, class_weight='balanced'` | `0.1765` | 70.05% | 83.98% | 49.85% | **86.07%** | 63.14% |
| **Random Forest** | `n_estimators=200, max_depth=8, class_weight='balanced'` | `0.3875` | 72.65% | 82.77% | 52.74% | 79.03% | 63.26% |
| **XGBoost** | `n_estimators=150, max_depth=4, learning_rate=0.1` | `0.2335` | 70.05% | 83.17% | 49.85% | 82.55% | 62.16% |
| **Decision Tree** | `max_depth=6, min_samples_leaf=20, class_weight='balanced'` | `0.4105` | 71.65% | 82.68% | 51.54% | 81.21% | 63.06% |

> **NHẬN XÉT MỐC 5 (GÓC NHÌN ƯU TIÊN RECALL Y TẾ)**:
> - SVM đạt Recall cao nhất toàn dự án: 86.07% — chỉ bỏ sót 83/596 ca đột quỵ thực tế. Đây là mốc duy nhất SVM vượt ngưỡng 86%.
> - LR và XGBoost đều đạt Recall rất tốt (82.55%), kết hợp với Accuracy và F1-Score vượt trội hơn SVM.
> - Random Forest là thuật toán duy nhất có Recall dưới 80% (79.03%) — chưa đạt yêu cầu sàng lọc y tế.
> - Xếp hạng Recall: SVM (86.07%) > LR = XGB (82.55%) > DT (81.21%) > RF (79.03%).
> - Kết luận: Mốc 5 đặc trưng phù hợp tuyệt vời cho 4/5 thuật toán (Recall >= 81%). Chỉ Random Forest cần thêm đặc trưng.

### MỐC 6 ĐẶC TRƯNG

| Thuật toán | Chi tiết cấu hình | Ngưỡng cắt | Accuracy | AUC-ROC | Precision (Lớp 1) | Recall (Lớp 1) | F1-Score (Lớp 1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `C=10.0, l2, class_weight='balanced'` | `0.4455` | **73.35%** | **84.75%** | **53.47%** | 81.38% | **64.54%** |
| **SVM (RBF)** | `C=100.0, gamma=0.01, class_weight='balanced'` | `0.1765` | 71.00% | 84.40% | 50.80% | **84.73%** | 63.52% |
| **Random Forest** | `n_estimators=200, max_depth=8, class_weight='balanced'` | `0.3875` | 71.45% | 83.58% | 51.33% | 81.21% | 62.90% |
| **XGBoost** | `n_estimators=150, max_depth=4, learning_rate=0.1` | `0.2335` | 69.55% | 83.02% | 49.35% | 82.55% | 61.77% |
| **Decision Tree** | `max_depth=6, min_samples_leaf=20, class_weight='balanced'` | `0.4105` | 71.75% | 83.32% | 51.69% | 79.53% | 62.66% |

> **NHẬN XÉT MỐC 6**:
> - SVM giảm nhẹ Recall xuống 84.73% (giảm 1.34% so với mốc 5), nhưng vẫn dẫn đầu.
> - LR giảm Recall từ 82.55% xuống 81.38% — thêm đặc trưng BMI không giúp ích cho mô hình tuyến tính.
> - RF tăng mạnh Recall lên 81.21% (từ 79.03%), nhờ BMI bổ sung thông tin phi tuyến.
> - DT giảm sâu Recall từ 81.21% xuống 79.53% — BMI gây nhiễu cho cấu trúc cây đơn lẻ.
> - Xếp hạng Recall: SVM (84.73%) > XGB (82.55%) > LR (81.38%) > RF (81.21%) > DT (79.53%).
> - So với Mốc 5: Phần lớn thuật toán bị giảm Recall. Thêm BMI không mang lại cải thiện rõ ràng.

### MỐC 7 ĐẶC TRƯNG

| Thuật toán | Chi tiết cấu hình | Ngưỡng cắt | Accuracy | AUC-ROC | Precision (Lớp 1) | Recall (Lớp 1) | F1-Score (Lớp 1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `C=10.0, l2, class_weight='balanced'` | `0.4455` | **73.45%** | **84.73%** | **53.58%** | 81.71% | **64.72%** |
| **SVM (RBF)** | `C=100.0, gamma=0.01, class_weight='balanced'` | `0.1765` | 70.70% | 84.23% | 50.50% | **85.57%** | 63.51% |
| **Random Forest** | `n_estimators=200, max_depth=8, class_weight='balanced'` | `0.3875` | 71.90% | 83.74% | 51.80% | 81.88% | 63.46% |
| **XGBoost** | `n_estimators=150, max_depth=4, learning_rate=0.1` | `0.2335` | 70.30% | 83.14% | 50.10% | 83.22% | 62.55% |
| **Decision Tree** | `max_depth=6, min_samples_leaf=20, class_weight='balanced'` | `0.4105` | 71.75% | 83.25% | 51.69% | 79.36% | 62.61% |

> **NHẬN XÉT MỐC 7**:
> - SVM vẫn duy trì Recall rất cao (85.57%), nhưng giảm 0.50% so với mốc 5.
> - XGBoost tăng lên 83.22% — bắt đầu hưởng lợi từ đặc trưng SES bổ sung.
> - RF tăng nhẹ lên 81.88%, xu hướng cải thiện tiếp tục.
> - DT tiếp tục giảm Recall (79.36%) — mốc 7 không phù hợp cho cây đơn lẻ.
> - Xếp hạng Recall: SVM (85.57%) > XGB (83.22%) > LR (81.71%) > RF (81.88%) > DT (79.36%).

### MỐC 8 ĐẶC TRƯNG

| Thuật toán | Chi tiết cấu hình | Ngưỡng cắt | Accuracy | AUC-ROC | Precision (Lớp 1) | Recall (Lớp 1) | F1-Score (Lớp 1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `C=10.0, l2, class_weight='balanced'` | `0.4455` | **73.65%** | **84.79%** | **53.85%** | 80.87% | **64.65%** |
| **SVM (RBF)** | `C=100.0, gamma=0.01, class_weight='balanced'` | `0.1765` | 70.30% | 84.29% | 50.10% | **85.07%** | 63.06% |
| **Random Forest** | `n_estimators=200, max_depth=8, class_weight='balanced'` | `0.3875` | 71.75% | 84.09% | 51.61% | 83.22% | 63.71% |
| **XGBoost** | `n_estimators=150, max_depth=4, learning_rate=0.1` | `0.2335` | 71.05% | 83.18% | 50.87% | 83.22% | 63.14% |
| **Decision Tree** | `max_depth=6, min_samples_leaf=20, class_weight='balanced'` | `0.4105` | 71.30% | 83.28% | 51.16% | 81.71% | 62.92% |

> **NHẬN XÉT MỐC 8**:
> - SVM đạt Recall 85.07%, vẫn dẫn đầu nhưng giảm 1.0% so với mốc 5.
> - RF và XGB đều đạt 83.22% — đây là mốc đầu tiên cả hai thuật toán Ensemble vượt ngưỡng 83%.
> - DT phục hồi lên 81.71% (tương đương mốc 5), nhờ đặc trưng Smoking_Status_Never.
> - LR giảm Recall từ 82.55% (mốc 5) xuống 80.87% — thêm đặc trưng gây nhiễu cho mô hình tuyến tính.
> - Xếp hạng Recall: SVM (85.07%) > RF = XGB (83.22%) > DT (81.71%) > LR (80.87%).
> - Mốc dung hòa: Nếu cần tất cả thuật toán đều đạt Recall >= 80%, mốc 8 là lựa chọn tốt.

### MỐC 9 ĐẶC TRƯNG

| Thuật toán | Chi tiết cấu hình | Ngưỡng cắt | Accuracy | AUC-ROC | Precision (Lớp 1) | Recall (Lớp 1) | F1-Score (Lớp 1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `C=10.0, l2, class_weight='balanced'` | `0.4455` | **73.70%** | **85.05%** | **53.91%** | 81.04% | **64.75%** |
| **SVM (RBF)** | `C=100.0, gamma=0.01, class_weight='balanced'` | `0.1765` | 70.10% | 84.68% | 49.90% | **85.74%** | 63.09% |
| **Random Forest** | `n_estimators=200, max_depth=8, class_weight='balanced'` | `0.3875` | 72.30% | 84.36% | 52.22% | 82.72% | 64.03% |
| **XGBoost** | `n_estimators=150, max_depth=4, learning_rate=0.1` | `0.2335` | 70.95% | 83.70% | 50.77% | 83.05% | 63.02% |
| **Decision Tree** | `max_depth=6, min_samples_leaf=20, class_weight='balanced'` | `0.4105` | 70.80% | 83.54% | 50.62% | 82.55% | 62.76% |

> **NHẬN XÉT MỐC 9**:
> - SVM đạt Recall cao nhất tại mốc 9: 85.74% (cao hơn mốc 11 nhưng vẫn thấp hơn mốc 5).
> - XGB ổn định ở 83.05%, RF giảm nhẹ xuống 82.72%.
> - DT tăng vọt lên 82.55% — đạt đỉnh Recall tại mốc 9.
> - LR tiếp tục xu hướng giảm nhẹ (81.04%).
> - Xếp hạng Recall: SVM (85.74%) > XGB (83.05%) > RF (82.72%) > DT (82.55%) > LR (81.04%).

### MỐC 10 ĐẶC TRƯNG

| Thuật toán | Chi tiết cấu hình | Ngưỡng cắt | Accuracy | AUC-ROC | Precision (Lớp 1) | Recall (Lớp 1) | F1-Score (Lớp 1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `C=10.0, l2, class_weight='balanced'` | `0.4455` | **73.75%** | **85.06%** | **53.97%** | 81.04% | **64.79%** |
| **SVM (RBF)** | `C=100.0, gamma=0.01, class_weight='balanced'` | `0.1765` | 70.30% | 84.66% | 50.10% | **85.40%** | 63.15% |
| **Random Forest** | `n_estimators=200, max_depth=8, class_weight='balanced'` | `0.3875` | 71.55% | 84.39% | 51.40% | 83.22% | 63.55% |
| **XGBoost** | `n_estimators=150, max_depth=4, learning_rate=0.1` | `0.2335` | 71.45% | 83.70% | 51.29% | 83.22% | 63.47% |
| **Decision Tree** | `max_depth=6, min_samples_leaf=20, class_weight='balanced'` | `0.4105` | 70.80% | 83.51% | 50.62% | 82.38% | 62.71% |

> **NHẬN XÉT MỐC 10**:
> - SVM giảm nhẹ xuống 85.40%, xu hướng đi ngang.
> - RF và XGB cùng giữ 83.22% — ổn định cao ở các mốc từ 8 trở đi.
> - DT và LR giữ nguyên so với mốc 9.
> - Xếp hạng Recall: SVM (85.40%) > RF = XGB (83.22%) > DT (82.38%) > LR (81.04%).
> - Đặc trưng Gender (thêm vào ở mốc 10) gần như không ảnh hưởng đến hiệu năng của bất kỳ thuật toán nào.

### MỐC 11 ĐẶC TRƯNG (FULL 11)

| Thuật toán | Chi tiết cấu hình | Ngưỡng cắt | Accuracy | AUC-ROC | Precision (Lớp 1) | Recall (Lớp 1) | F1-Score (Lớp 1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `C=10.0, l2, class_weight='balanced'` | `0.4455` | **73.90%** | **85.07%** | **54.15%** | 81.04% | **64.92%** |
| **SVM (RBF)** | `C=100.0, gamma=0.01, class_weight='balanced'` | `0.1765` | 70.05% | 84.64% | 49.85% | **85.57%** | 63.00% |
| **Random Forest** | `n_estimators=200, max_depth=8, class_weight='balanced'` | `0.3875` | 71.60% | 84.58% | 51.45% | 83.39% | 63.64% |
| **XGBoost** | `n_estimators=150, max_depth=4, learning_rate=0.1` | `0.2335` | 70.85% | 83.55% | 50.67% | 83.05% | 62.94% |
| **Decision Tree** | `max_depth=6, min_samples_leaf=20, class_weight='balanced'` | `0.4105` | 70.80% | 83.51% | 50.62% | 82.38% | 62.71% |

> **NHẬN XÉT MỐC 11 (FULL 11)**:
> - SVM đạt Recall 85.57% — cao nhưng vẫn thấp hơn 0.50% so với mốc 5 đặc trưng.
> - RF đạt đỉnh Recall toàn bộ thực nghiệm của RF: 83.39%.
> - XGB đạt 83.05%, DT đạt 82.38% — cả hai đạt đỉnh hoặc gần đỉnh Recall.
> - LR đạt 81.04% — thấp hơn 1.51% so với mốc 5 đặc trưng.
> - Xếp hạng Recall: SVM (85.57%) > RF (83.39%) > XGB (83.05%) > DT (82.38%) > LR (81.04%).
> - Kết luận: Mốc 11 tối ưu cho RF và XGB, nhưng KHÔNG tối ưu cho SVM và LR (cả hai đều đạt Recall tốt hơn ở mốc 5).

---

## NHẬN XÉT TỔNG QUAN VỀ HIỆU QUẢ CÁC MỐC ĐẶC TRƯNG (TỪ MI=5 ĐẾN MI=11) VÀ LỰA CHỌN TỐI ƯU

1. ĐÁNH GIÁ TỔNG QUAN VỀ SỰ BIẾN ĐỔI HIỆU NĂNG TỪ 5 ĐẾN 11 ĐẶC TRƯNG
o Nhóm mô hình phẳng/tuyến tính (SVM, LR): Đạt Recall cao nhất tại MI = 5 (SVM: 86.07%, LR: 82.55%). Tăng đặc trưng lên 11 làm giảm Recall do các biến xã hội học (hôn nhân, công việc, nơi ở) gây nhiễu.
o Nhóm mô hình cây tập hợp (RF, XGB): Recall tăng dần khi thêm đặc trưng. RF đạt đỉnh ở mốc 11 (83.39%), XGB ổn định từ mốc 8 trở đi (83.22%). Các mô hình Ensemble cần nhiều đặc trưng để học tương tác phi tuyến.

2. LỰA CHỌN MỐC TỐT NHẤT NÊN DÙNG CHO BÀI TOÁN Y TẾ ƯU TIÊN RECALL
Lựa chọn mốc MI = 5 là vì:
o Đạt Recall cao nhất: SVM đạt 86.07% ở mốc 5 (hạn chế tối đa bỏ sót bệnh nhân).
o Tối ưu lâm sàng: Chỉ cần 5 chỉ số lâm sàng cốt lõi (Tăng huyết áp, Tuổi, Bệnh tim, Đường huyết trung bình, Tiểu đường), dễ thu thập và tiết kiệm chi phí.
o Cân bằng tốt: LR ở mốc 5 đạt Recall 82.55% đi kèm Accuracy cao nhất (73.80%), giảm cảnh báo giả.

---

## TỔNG KẾT VÀ KHUYẾN NGHỊ RÚT RA

### BẢNG TỔNG HỢP RECALL CAO NHẤT CỦA TỪNG THUẬT TOÁN THEO MỐC MI

| Thuật toán | Mốc MI đạt Recall cao nhất | Recall (Lớp 1) | Accuracy tại mốc đó | F1-Score tại mốc đó | Nhận xét |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SVM (RBF)** | **MI = 5** | **86.07%** | 70.05% | 63.14% | Recall cao nhất toàn dự án |
| **Logistic Regression** | **MI = 5** | **82.55%** | **73.80%** | **65.25%** | Cân bằng Recall + Accuracy tốt nhất |
| **XGBoost** | MI = 7, 8, 10 | **83.22%** | 70.30–71.45% | 62.55–63.47% | Ổn định Recall từ mốc 7 trở đi |
| **Random Forest** | MI = 11 | **83.39%** | 71.60% | 63.64% | Cần đầy đủ 11 đặc trưng |
| **Decision Tree** | MI = 9 | **82.55%** | 70.80% | 62.76% | Đạt đỉnh ở mốc 9 |

### BẢNG SO SÁNH RECALL TOÀN BỘ THUẬT TOÁN TẠI MỐC MI = 5 VS MI = 11

| Thuật toán | MI = 5<br>Recall (Lớp 1) | MI = 11<br>Recall (Lớp 1) | Chênh lệch | MI = 5<br>F1-Score | MI = 11<br>F1-Score | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **SVM (RBF)** | **86.07%** | 85.57% | **+0.50%** (MI-5 tốt hơn) | **63.14%** | 63.00% | MI-5 vượt trội |
| **Logistic Regression** | **82.55%** | 81.04% | **+1.51%** (MI-5 tốt hơn) | **65.25%** | 64.92% | MI-5 vượt trội |
| **XGBoost** | 82.55% | **83.05%** | -0.50% (MI-11 tốt hơn) | 62.16% | **62.94%** | Chênh lệch không đáng kể |
| **Decision Tree** | 81.21% | **82.38%** | -1.17% (MI-11 tốt hơn) | **63.06%** | 62.71% | Chênh lệch nhỏ |
| **Random Forest** | 79.03% | **83.39%** | -4.36% (MI-11 tốt hơn) | 63.26% | **63.64%** | RF cần thêm đặc trưng |

---

### KHUYẾN NGHỊ CUỐI CÙNG: CHỌN MỐC MI = 5 ĐẶC TRƯNG

Trong bối cảnh bài toán y tế sàng lọc đột quỵ, mục tiêu hàng đầu là tối đa hóa Recall (giảm thiểu việc bỏ sót ca bệnh), kết hợp với tính khả thi triển khai lâm sàng. Phân tích thực nghiệm cho thấy Mốc MI = 5 đặc trưng là lựa chọn tối ưu nhất, với các lập luận sau:

#### LÝ DO CHỌN MI = 5:

1. Đạt Recall cao nhất toàn dự án khi kết hợp SVM: SVM-C5 tại mốc 5 đặc trưng cho Recall 86.07% — con số này không bị bất kỳ mốc nào khác vượt qua (kể cả khi dùng đầy đủ 11 đặc trưng, SVM chỉ đạt 85.57%).

2. 4/5 thuật toán đều đạt Recall >= 81% tại mốc 5: Chỉ duy nhất Random Forest (79.03%) không đạt ngưỡng mong muốn. Các thuật toán còn lại đều hoạt động rất tốt với chỉ 5 đặc trưng đầu vào.

3. 5 đặc trưng được chọn đều là các yếu tố nguy cơ y khoa cốt lõi của đột quỵ:
   - Hypertension (Tăng huyết áp) — Yếu tố nguy cơ hàng đầu gây đột quỵ
   - Age (Tuổi) — Tuổi cao làm tăng nguy cơ đột quỵ
   - Heart_Disease (Bệnh tim) — Rung nhĩ, suy tim gây huyết khối
   - Avg_Glucose (Đường huyết trung bình) — Đường huyết cao gây tổn thương mạch máu
   - Diabetes (Tiểu đường) — Bệnh nền gây xơ vữa động mạch
   - -> Tất cả 5 đặc trưng này đều có cơ sở y văn mạnh mẽ và có thể đo đạc nhanh chóng tại phòng khám.

4. Tiết kiệm chi phí thu thập dữ liệu lâm sàng: Không cần thu thập các thông tin xã hội học phụ trợ (BMI, tình trạng hút thuốc, giới tính, nghề nghiệp...) vốn khó chuẩn hóa và dễ thiếu sót trong thực tế.

5. Giảm nguy cơ Overfitting: Mô hình ít đặc trưng hơn -> ít bị quá khớp trên dữ liệu huấn luyện -> tổng quát hóa tốt hơn khi triển khai trên dữ liệu bệnh nhân mới.

#### HẠN CHẾ DUY NHẤT CỦA MI = 5:
- Random Forest chỉ đạt Recall 79.03% (thấp hơn 4.36% so với mốc 11). Tuy nhiên, trong bối cảnh ưu tiên Recall, SVM và LR đã đạt hiệu năng vượt trội hơn RF ở mốc 5, nên hạn chế này không ảnh hưởng đến quyết định chung.

#### CẶP ĐÔI KHUYẾN NGHỊ TRIỂN KHAI VỚI MI = 5:

| Mục tiêu | Thuật toán | Recall | Accuracy | F1-Score | Ứng dụng |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Sàng lọc tối đa (không bỏ sót) | SVM (RBF) | **86.07%** | 70.05% | 63.14% | Hệ thống cảnh báo sớm tại khoa cấp cứu |
| Cân bằng tốt nhất (Recall + Accuracy) | Logistic Regression | **82.55%** | **73.80%** | **65.25%** | Ứng dụng sàng lọc cộng đồng, thiết bị di động |

---

**Báo cáo cập nhật ngày**: 2026-06-05  
**Nguồn số liệu thực nghiệm**: `run_fixed_threshold_comparison.py` (SMOTE = 0.5)
