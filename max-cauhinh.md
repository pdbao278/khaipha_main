# 🛠️ CẤU HÌNH CHI TIẾT CÁC MÔ HÌNH HỌC MÁY (ALGORITHM CONFIGURATIONS)

Tài liệu này trích xuất và tổng hợp chi tiết cấu hình siêu tham số (hyperparameters), ngưỡng quyết định (threshold), và hiệu năng của **5 thuật toán chính** được sử dụng trong báo cáo đối sánh [so_sanh_full11_vs_mi5.md](file:///e:/kpdl_main/so_sanh_full11_vs_mi5.md).

Mục tiêu là cung cấp một tài liệu tham chiếu nhanh, chuẩn xác về mặt kỹ thuật cho quá trình huấn luyện và tái lập kết quả mô hình.

---

## 📊 BẢNG TỔNG HỢP CẤU HÌNH NHANH

| Thuật toán | Mã Cấu hình | Siêu tham số chính | Ngưỡng (Full 11) | Ngưỡng động (MI 5) | Trạng thái (Recall MI 5) |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Logistic Regression** | **LR-C4** | `C=10.0`, `class_weight='balanced'` | `0.4455` | `0.4480` | **ĐẠT > 80%** (`82.55%`) |
| **SVM (RBF)** | **SVM-C5** | `C=100.0`, `gamma=0.01`, `class_weight='balanced'` | `0.1765` | `0.2155` | **ĐẠT > 80%** (`80.54%`) |
| **Random Forest** | **RF-C3** | `max_depth=8`, `n_estimators=200`, `class_weight='balanced'` | `0.3875` | `0.3920` | Chưa đạt (`78.86%`) |
| **XGBoost** | **XGB-C3** | `max_depth=4`, `learning_rate=0.1`, `n_estimators=150` | `0.2335` | `0.2510` | Chưa đạt (`79.36%`) |
| **Decision Tree** | **DT-C3** | `max_depth=6`, `min_samples_leaf=20`, `class_weight='balanced'` | `0.4105` | `0.4315` | **ĐẠT > 80%** (`80.20%`) |

---

## 🔍 CHI TIẾT CẤU HÌNH TỪNG THUẬT TOÁN

### 1. Logistic Regression (Mã: `LR-C4`)
* **Siêu tham số khởi tạo (Python):**
  ```python
  LogisticRegression(C=10.0, class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42)
  ```
* **Chiến lược ngưỡng & Hiệu năng:**
  * **Ngưỡng cố định (Fixed Threshold):** `0.4455`
  * **Ngưỡng động tối ưu (Dynamic Threshold):** `0.4480`
  * **Hiệu năng trên tập Full 11 Features:**
    * Recall: **81.04%** | Precision: **54.15%** | F1-Score: **64.92%** | AUC-ROC: **85.07%**
  * **Hiệu năng trên tập MI 5 Features (Ngưỡng động):**
    * Recall: **82.55%** | Precision: **54.19%** | F1-Score: **65.43%** | AUC-ROC: **84.63%**

---

### 2. SVM RBF Kernel (Mã: `SVM-C5`)
* **Siêu tham số khởi tạo (Python):**
  ```python
  SVC(C=100.0, kernel='rbf', gamma=0.01, class_weight='balanced', probability=True, random_state=42)
  ```
* **Chiến lược ngưỡng & Hiệu năng:**
  * **Ngưỡng cố định (Fixed Threshold):** `0.1765`
  * **Ngưỡng động tối ưu (Dynamic Threshold):** `0.2155`
  * **Hiệu năng trên tập Full 11 Features:**
    * Recall: **85.57%** | Precision: **49.85%** | F1-Score: **63.00%** | AUC-ROC: **84.64%**
  * **Hiệu năng trên tập MI 5 Features (Ngưỡng động):**
    * Recall: **80.54%** | Precision: **54.98%** | F1-Score: **65.35%** | AUC-ROC: **83.98%**

---

### 3. Random Forest (Mã: `RF-C3`)
* **Siêu tham số khởi tạo (Python):**
  ```python
  RandomForestClassifier(max_depth=8, n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
  ```
* **Chiến lược ngưỡng & Hiệu năng:**
  * **Ngưỡng cố định (Fixed Threshold):** `0.3875`
  * **Ngưỡng động tối ưu (Dynamic Threshold):** `0.3920`
  * **Hiệu năng trên tập Full 11 Features:**
    * Recall: **84.06%** | Precision: **50.86%** | F1-Score: **63.38%** | AUC-ROC: **84.50%**
  * **Hiệu năng trên tập MI 5 Features (Ngưỡng động):**
    * Recall: **78.86%** | Precision: **53.05%** | F1-Score: **63.43%** | AUC-ROC: **82.77%**

---

### 4. XGBoost (Mã: `XGB-C3`)
* **Siêu tham số khởi tạo (Python):**
  ```python
  XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=150, eval_metric='logloss', random_state=42, n_jobs=-1)
  ```
* **Chiến lược ngưỡng & Hiệu năng:**
  * **Ngưỡng cố định (Fixed Threshold):** `0.2335`
  * **Ngưỡng động tối ưu (Dynamic Threshold):** `0.2510`
  * **Hiệu năng trên tập Full 11 Features:**
    * Recall: **83.05%** | Precision: **50.67%** | F1-Score: **62.94%** | AUC-ROC: **83.55%**
  * **Hiệu năng trên tập MI 5 Features (Ngưỡng động):**
    * Recall: **79.36%** | Precision: **51.75%** | F1-Score: **62.65%** | AUC-ROC: **83.17%**

---

### 5. Decision Tree (Mã: `DT-C3`)
* **Siêu tham số khởi tạo (Python):**
  ```python
  DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight='balanced', random_state=42)
  ```
* **Chiến lược ngưỡng & Hiệu năng:**
  * **Ngưỡng cố định (Fixed Threshold):** `0.4105`
  * **Ngưỡng động tối ưu (Dynamic Threshold):** `0.4315`
  * **Hiệu năng trên tập Full 11 Features:**
    * Recall: **82.38%** | Precision: **50.62%** | F1-Score: **62.71%** | AUC-ROC: **83.51%**
  * **Hiệu năng trên tập MI 5 Features (Ngưỡng động):**
    * Recall: **80.20%** | Precision: **53.29%** | F1-Score: **64.03%** | AUC-ROC: **82.68%**

---

## 📌 THÔNG TIN VỀ DỮ LIỆU ĐẦU VÀO
* **Bộ đầy đủ 11 đặc trưng (Full 11 Features):** Toàn bộ các đặc trưng ban đầu sau khi xử lý.
* **Bộ rút gọn 5 đặc trưng Mutual Information (MI Top 5 Features):** `Age` (Tuổi tác), `Hypertension` (Tăng huyết áp), `Heart_Disease` (Bệnh tim), `Avg_Glucose` (Lượng đường huyết trung bình), `Diabetes` (Tiểu đường).
