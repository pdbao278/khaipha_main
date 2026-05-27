# 🎯 CẤU HÌNH CÁC MÔ HÌNH CHÍNH (MAIN MODELS CONFIGURATIONS)

Tài liệu này lưu trữ thông tin chi tiết về cấu hình tối ưu của các thuật toán học máy được lựa chọn từ biểu đồ báo cáo thực nghiệm thực tế (**Full 11 Features**), hướng tới mục tiêu tối ưu hóa **Recall $\ge 80\%$** trong khi duy trì F1-Score và Precision ở mức cao nhất để hạn chế báo động giả.

---

## 📊 BẢNG TỔNG HỢP CẤU HÌNH CHỌN LỌC

| Thuật toán | Mã Cấu hình | Siêu tham số chính | Ngưỡng (Threshold) | Recall (Lớp 1) | Precision (Lớp 1) | F1-Score (Lớp 1) | AUC-ROC |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **LR-C4** | $C=10.0$, balanced | **0.4455** | **81.04%** | **54.15%** | **64.92%** | **85.07%** |
| **SVM (RBF)** | **SVM-C5** | $C=100.0$, gamma=0.01, balanced | **0.1765** | **85.57%** | **49.85%** | **63.00%** | **84.64%** |
| **Random Forest** | **RF-C3** | depth=8, n=200, balanced | **0.3875** | **84.06%** | **50.86%** | **63.38%** | **84.50%** |
| **XGBoost** | **XGB-C3** | depth=4, lr=0.1, n=150 | **0.2335** | **83.05%** | **50.67%** | **62.94%** | **83.55%** |
| **Decision Tree** | **DT-C3** | depth=6, leaf=20, balanced | **0.4105** | **82.38%** | **50.62%** | **62.71%** | **83.51%** |

> [!NOTE]
> Chỉ số F1-Score của **Decision Tree (DT-C3)** trên biểu đồ thực nghiệm gốc bị ghi nhầm thành `61.71%`. Giá trị thực tế tính toán theo công thức chính xác là **62.71%** (đã được cập nhật chuẩn xác trong bảng trên).

---

## 🔍 CHI TIẾT CẤU HÌNH TỪNG THUẬT TOÁN

### 1. Logistic Regression (Mã: `LR-C4`)
* **Siêu tham số:**
  ```python
  LogisticRegression(C=10.0, class_weight='balanced', random_state=42)
  ```
* **Ngưỡng phân loại tối ưu (Threshold):** `0.4455`
* **Đánh giá hiệu năng:**
  * Recall: **81.04%**
  * Precision: **54.15%**
  * F1-Score: **64.92%**
  * AUC-ROC: **85.07%**

### 2. Support Vector Machine - RBF Kernel (Mã: `SVM-C5`)
* **Siêu tham số:**
  ```python
  SVC(C=100.0, kernel='rbf', gamma=0.01, class_weight='balanced', probability=True, random_state=42)
  ```
* **Ngưỡng phân loại tối ưu (Threshold):** `0.1765`
* **Đánh giá hiệu năng:**
  * Recall: **85.57%** (Cao nhất trong nhóm mô hình chính)
  * Precision: **49.85%**
  * F1-Score: **63.00%**
  * AUC-ROC: **84.64%**

### 3. Random Forest (Mã: `RF-C3`)
* **Siêu tham số:**
  ```python
  RandomForestClassifier(max_depth=8, n_estimators=200, class_weight='balanced', random_state=42)
  ```
* **Ngưỡng phân loại tối ưu (Threshold):** `0.3875`
* **Đánh giá hiệu năng:**
  * Recall: **84.06%**
  * Precision: **50.86%**
  * F1-Score: **63.38%**
  * AUC-ROC: **84.50%**

### 4. XGBoost (Mã: `XGB-C3`)
* **Siêu tham số:**
  ```python
  XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=150, random_state=42)
  ```
* **Ngưỡng phân loại tối ưu (Threshold):** `0.2335`
* **Đánh giá hiệu năng:**
  * Recall: **83.05%**
  * Precision: **50.67%**
  * F1-Score: **62.94%**
  * AUC-ROC: **83.55%**

### 5. Decision Tree (Mã: `DT-C3`)
* **Siêu tham số:**
  ```python
  DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight='balanced', random_state=42)
  ```
* **Ngưỡng phân loại tối ưu (Threshold):** `0.4105`
* **Đánh giá hiệu năng:**
  * Recall: **82.38%**
  * Precision: **50.62%**
  * F1-Score: **62.71%** (Giá trị thực tế trên tập Test)
  * AUC-ROC: **83.51%**
