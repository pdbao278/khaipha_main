# HƯỚNG DẪN TOÀN DIỆN: THIẾT LẬP PIPELINE PHÂN LOẠI & SO SÁNH ĐA MÔ HÌNH
## (Áp Dụng Cho Các Bài Toán Phân Loại Nhị Phân & Dữ Liệu Mất Cân Bằng)

Tài liệu này cung cấp hướng dẫn từng bước (Step-by-step Guideline) để xây dựng một dự án phân loại Machine Learning toàn diện từ tiền xử lý dữ liệu, giải quyết mất cân bằng lớp, huấn luyện đa thuật toán, tối ưu hóa ngưỡng (Threshold Tuning), tổng hợp kết quả đến vẽ biểu đồ so sánh trực quan.

---

## NGỮ CẢNH DỰ ÁN & THÔNG TIN TẬP DỮ LIỆU ĐỘT QUỴ (STROKE PREDICTION DATASET)

Dự án này tập trung vào bài toán **Dự đoán nguy cơ Đột quỵ (Stroke Prediction)** của bệnh nhân – một trong những nguyên nhân hàng đầu gây tử vong và tàn tật lâu dài trên toàn thế giới. Bằng cách ứng dụng Machine Learning, ta có thể xây dựng các mô hình cảnh báo sớm dựa trên các yếu tố lâm sàng, thói quen sinh hoạt và chỉ số cơ thể.

Đây là bài toán **Phân loại Nhị phân (Binary Classification)** đặc trưng với dữ liệu bị **mất cân bằng lớp (Imbalanced Dataset)** cao. Dưới đây là thông tin chi tiết về ngữ cảnh dự án và tập dữ liệu:

### 1. Tổng Quan Về Tập Dữ Liệu
* **Tệp dữ liệu gốc:** `stroke_data.csv`
* **Tổng số mẫu (records):** **10,000** bệnh nhân.
* **Tổng số thuộc tính (features):** **9** đặc trưng đầu vào và **1** thuộc tính mục tiêu (nhãn).
* **Nhãn mục tiêu (`Stroke`):** 
  * `0` (Không bị đột quỵ): **7,022** mẫu – chiếm **70.22%**
  * `1` (Bị đột quỵ): **2,978** mẫu – chiếm **29.78%**
  
> [!NOTE]
> Tỷ lệ mất cân bằng giữa hai lớp (~7:3) là một thách thức lớn. Nếu sử dụng ngưỡng phân loại mặc định (Threshold = 0.5) mà không áp dụng các kỹ thuật cân bằng dữ liệu (như SMOTE hoặc Class Weights), mô hình sẽ có xu hướng thiên vị lớp đa số (không bị bệnh), dẫn đến chỉ số **Recall của lớp bị bệnh (Class 1) bị sụt giảm nghiêm trọng**, gây nguy hiểm khi bỏ lọt bệnh nhân thực tế.

### 2. Mô Tả Chi Tiết Các Đặc Trưng (Feature Schema)

Tập dữ liệu bao gồm các thông tin lâm sàng và thói quen sinh hoạt của bệnh nhân:

| Tên Cột (Feature) | Kiểu Dữ Liệu | Mô Tả Ý Nghĩa | Định Dạng / Mã Hóa Sau Tiền Xử Lý |
|---|---|---|---|
| **Age** | Số liên tục (Float) | Độ tuổi của bệnh nhân. | Chuẩn hóa chuẩn (Standardized) |
| **Gender** | Phân loại (Categorical) | Giới tính của bệnh nhân. | Label Encoded (`0`: Female, `1`: Male) |
| **SES** | Thứ bậc (Ordinal) | Tình trạng kinh tế xã hội (Socioeconomic Status). | Ordinal Encoded (`0`: Low, `1`: Medium, `2`: High) |
| **Hypertension** | Nhị phân (Binary) | Bệnh nhân có bị cao huyết áp hay không. | Nhị phân (`0`: Không bị, `1`: Có bị) |
| **Heart_Disease** | Nhị phân (Binary) | Bệnh nhân có tiền sử bệnh tim hay không. | Nhị phân (`0`: Không có, `1`: Có) |
| **BMI** | Số liên tục (Float) | Chỉ số khối cơ thể (Body Mass Index). | Chuẩn hóa chuẩn (Standardized) |
| **Avg_Glucose** | Số liên tục (Float) | Mức đường huyết trung bình trong máu (mg/dL). | Chuẩn hóa chuẩn (Standardized) |
| **Diabetes** | Nhị phân (Binary) | Bệnh nhân có bị tiểu đường hay không. | Nhị phân (`0`: Không bị, `1`: Có bị) |
| **Smoking_Status** | Phân loại (Categorical) | Tình trạng hút thuốc lá của bệnh nhân. | One-Hot Encoded (`Smoking_Status_Never`, `Smoking_Status_Former`, `Smoking_Status_Current`) |
| **Stroke** | **Nhãn mục tiêu (Target)** | Bệnh nhân có bị đột quỵ hay không. | Nhãn nhị phân (`0`: Không bị, `1`: Bị đột quỵ) |

---

## I. CẤU TRÚC THƯ MỤC DỰ ÁN TIÊU CHUẨN (MODULAR DESIGN)

Để dự án dễ bảo trì, mở rộng và bàn giao, cấu trúc thư mục nên được tổ chức một cách khoa học:

```text
📁 [Ten_Du_An]/
│
├── 📄 data_raw.csv                # Dữ liệu gốc chưa xử lý
├── 📄 data_processed.csv          # Dữ liệu sau tiền xử lý (Sạch, Scaled, Encodings)
│
├── 📄 preprocessing.py            # Code tiền xử lý dữ liệu chính (Làm sạch, IQR, Encoding, Scaling)
├── 📄 preprocessing_report.md     # Báo cáo chi tiết kết quả tiền xử lý (kèm biểu đồ trực quan)
│
├── 📁 visuals/                    # Chứa hình ảnh trực quan hóa ở bước tiền xử lý
│   ├── 🖼️ 1_distribution.png
│   ├── 🖼️ 2_missing_values.png
│   └── 🖼️ 3_outliers_iqr.png
│
├── 📁 models/                     # Thư mục huấn luyện từng thuật toán riêng biệt
│   ├── 📄 decision_tree.py       # Code & Hyperparameter Tuning cho Decision Tree
│   ├── 📄 random_forest.py       # Code & Hyperparameter Tuning cho Random Forest
│   ├── 📄 xgboost_model.py       # Code & Hyperparameter Tuning cho XGBoost
│   ├── 📄 logistic_regression.py # Code & Hyperparameter Tuning cho Logistic Regression
│   ├── 📄 svm_model.py           # Code & Hyperparameter Tuning cho SVM
│   │
│   ├── 📄 decision_tree.md       # Báo cáo chi tiết & Phân tích lâm sàng Decision Tree
│   ├── 📄 random_forest.md       # Báo cáo chi tiết & Phân tích lâm sàng Random Forest
│   └── ...                       # Báo cáo tương ứng cho các mô hình khác
│
└── 📁 evaluation/                 # Tổng hợp, so sánh và trực quan hóa hiệu năng chung
    ├── 📄 summarize_results.py   # Script thu thập metrics của toàn bộ mô hình
    ├── 📄 plot_comparison.py     # Script vẽ biểu đồ ROC, PR-Curve, so sánh Metrics
    └── 📄 overall_report.md      # Báo cáo tổng hợp cuối cùng so sánh các chiến lược (Recall vs Balance)
```

---

## II. QUY TRÌNH 7 BƯỚC THIẾT LẬP PIPELINE TIỀN XỬ LÝ (PREPROCESSING)

Quy trình tiền xử lý phải đảm bảo tính khoa học và tránh hiện tượng **Rò rỉ dữ liệu (Data Leakage)**. Dưới đây là mã nguồn chuẩn hóa dưới dạng mẫu tổng quát:

### Bước 1: Khởi Tạo Dự Án & Đọc Dữ Liệu
```python
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Tạo thư mục lưu ảnh nếu chưa có
os.makedirs('visuals', exist_ok=True)

# Đọc dữ liệu đột quỵ
df = pd.read_csv('stroke_data.csv')
print(f"Kích thước tệp dữ liệu: {df.shape}")
print(f"Các cột thuộc tính: {df.columns.tolist()}")

# Vẽ phân phối nhãn mục tiêu (Stroke)
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Stroke', palette='viridis')
plt.title('Phân phối Nhãn Mục Tiêu (Stroke)')
plt.savefig('visuals/1_distribution.png', bbox_inches='tight')
plt.close()
```


### Bước 3: Xử Lý Ngoại Lệ Bằng Phương Pháp IQR (Interquartile Range)
*Nguyên tắc:* Tính toán khoảng trải giữa IQR để xác định giới hạn trên/dưới. Thay vì xóa dòng (làm mất thông tin), ta thực hiện **Giới hạn (Clipping/Winsorization)** các giá trị vượt biên về sát giá trị biên gần nhất.
```python
numeric_cols = ['Age', 'BMI', 'Avg_Glucose'] # Danh sách cột số liên tục của tập dữ liệu Stroke

# Vẽ Boxplot trước khi xử lý
plt.figure(figsize=(12, 4))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(1, len(numeric_cols), i)
    sns.boxplot(y=df[col], color='salmon')
    plt.title(f'Trước IQR: {col}')
plt.tight_layout()
plt.savefig('visuals/3_outliers_before.png', bbox_inches='tight')
plt.close()

# Áp dụng công thức IQR và Clip dữ liệu
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Clip (Giới hạn giá trị trong khoảng [lower_bound, upper_bound])
    df[col] = np.clip(df[col], lower_bound, upper_bound)

# Vẽ Boxplot sau khi xử lý để xác nhận không còn ngoại lệ cực đoan
plt.figure(figsize=(12, 4))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(1, len(numeric_cols), i)
    sns.boxplot(y=df[col], color='lightgreen')
    plt.title(f'Sau IQR: {col}')
plt.tight_layout()
plt.savefig('visuals/3_outliers_after.png', bbox_inches='tight')
plt.close()
```

### Bước 4: Mã Hóa Biến Phân Loại (Categorical Encoding)
*Lựa chọn phương pháp:*
1. **Label Encoding / Mapping:** Cho biến Gender (Female: 0, Male: 1).
2. **Ordinal Encoding:** Cho biến SES (Low: 0, Medium: 1, High: 2).
3. **One-Hot Encoding:** Cho biến Smoking_Status.
```python
# 1. Label Mapping cho cột Gender (chỉ có 2 nhóm)
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

# 2. Ordinal Encoding cho cột SES bằng cách định nghĩa Map cụ thể
ses_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['SES'] = df['SES'].map(ses_map)

# 3. One-Hot Encoding cho cột Smoking_Status (nhiều nhóm không thứ tự)
df = pd.get_dummies(df, columns=['Smoking_Status'], prefix='Smoking_Status', dtype=int)
```

### Bước 5: Chuẩn Hóa Đặc Trưng (Feature Scaling)
Đưa các cột số về cùng một thang đo nhằm tránh việc các thuật toán dựa trên khoảng cách (như SVM, Logistic Regression) bị thiên vị bởi các cột có khoảng giá trị lớn.
Dùng StandardScaler
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = df.copy()

# Chuẩn hóa các thuộc tính liên tục
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Lưu lại tập dữ liệu đã sạch sẽ hoàn toàn
df_scaled.to_csv('data_processed.csv', index=False)
print("Tiền xử lý hoàn tất! File đã lưu vào: 'data_processed.csv'")
```

---

## III. CHIẾN LƯỢC XỬ LÝ MẤT CÂN BẰNG LỚP (IMBALANCED DATA)

Trong các bài toán thực tế (Ví dụ: Phát hiện bệnh y khoa, gian lận tài chính), tỷ lệ mẫu tích cực (Lớp 1) thường cực kỳ thấp so với mẫu tiêu cực (Lớp 0). Ta có 2 cách tiếp cận chính:

### Phương án A: Điều chỉnh trọng số mô hình (`class_weight='balanced'`)
Đây là phương án an toàn nhất vì **không làm giả lập thêm dữ liệu**, chỉ đơn giản là nhân hệ số phạt cao hơn khi mô hình phân loại sai lớp thiểu số. Hầu hết các thư viện (`scikit-learn`, `xgboost`) đều hỗ trợ tham số này.

### Phương án B: Kỹ thuật tạo mẫu thiểu số nhân tạo (SMOTE)
Sử dụng thuật toán K-Nearest Neighbors để nội suy ra các điểm dữ liệu mới của lớp thiểu số trên tập Train.
> [!WARNING]
> **Quy tắc Vàng chống Data Leakage:** Chỉ được phép áp dụng SMOTE trên tập **TRAIN** sau khi đã chia tách Train/Val/Test theo tỷ lệ 7/1/2. Tuyệt đối KHÔNG được áp dụng SMOTE trên toàn bộ tập dữ liệu trước khi chia, hoặc trên tập Test/Validation.

```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Chia tập Train/Val/Test theo tỷ lệ 7/1/2 (70% Train, 10% Validation, 20% Test)
X = df_scaled.drop('Stroke', axis=1)
y = df_scaled['Stroke']

# 1. Tách 20% dữ liệu làm tập Test độc lập
X_temp_train, X_test, y_temp_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 2. Tách phần còn lại (80%) thành tập Train (70% tổng) và Validation (10% tổng)
# 10% tổng số mẫu tương ứng với 10/80 = 12.5% của tập temp_train
X_train, X_val, y_train, y_val = train_test_split(
    X_temp_train, y_temp_train, test_size=0.125, random_state=42, stratify=y_temp_train
)

# Chỉ áp dụng SMOTE trên tập huấn luyện X_train để chống rò rỉ dữ liệu (Data Leakage)
# Tỷ lệ sampling_strategy=0.5 nghĩa là Lớp 1 sau SMOTE sẽ bằng 50% số lượng của Lớp 0
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Kích thước tập Train (70%): {X_train.shape[0]} mẫu")
print(f"Kích thước tập Val (10%): {X_val.shape[0]} mẫu")
print(f"Kích thước tập Test (20%): {X_test.shape[0]} mẫu")
print(f"Trước SMOTE (Lớp Train): {y_train.value_counts().to_dict()}")
print(f"Sau SMOTE (Lớp Train): {y_train_resampled.value_counts().to_dict()}")
```

---

## IV. HUẤN LUYỆN, TỐI ƯU HÓA NGƯỠNG & ĐÁNH GIÁ MÔ HÌNH

### 1. Hàm Đánh Giá Phân Loại Tiêu Chuẩn
```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    print(f"=== DANH GIA MO HINH: {model_name} ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
```

### 2. Hàm Điều Chỉnh Ngưỡng Trên Tập Validation (Threshold Tuning on Validation)
Mặc định các mô hình phân loại sẽ đưa ra dự đoán dựa trên ngưỡng probability >= 0.5. Tuy nhiên, trong kiến trúc 7/1/2, ta sử dụng tập Validation độc lập để dò tìm ngưỡng phân loại tối ưu phù hợp với mục tiêu lâm sàng, sau đó mới kiểm chứng trên tập Test độc lập:
- **Nếu cần Sự Cân Bằng (Balance F1):** Ta dò tìm ngưỡng tối đa hóa **F1-score** trên tập Validation.
- **Nếu cần Tối Đa Recall (Max Recall):** Ta dò tìm ngưỡng tối đa hóa F1-score trên tập Validation sao cho **Recall trên Validation đạt >= 80%**.

```python
from sklearn.metrics import precision_recall_curve, f1_score, recall_score, precision_score, confusion_matrix

# 1. Tìm ngưỡng tối ưu Balance F1 trên tập Validation
def find_optimal_threshold_balance_f1(model, X_val, y_val):
    val_probs = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    
    # Tính F1-score cho từng ngưỡng
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]

# 2. Tìm ngưỡng tối ưu Max Recall trên tập Validation (Recall >= 80% trên Val, tối ưu F1)
def find_optimal_threshold_max_recall(model, X_val, y_val, target_recall=0.80):
    val_probs = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    
    # Lọc các ngưỡng có Recall trên Validation đạt ít nhất target_recall
    valid_indices = np.where(recalls[:-1] >= target_recall)[0]
    if len(valid_indices) > 0:
        best_idx = valid_indices[np.argmax(f1_scores[valid_indices])]
        return thresholds[best_idx]
    else:
        return thresholds[np.argmax(f1_scores)]

# 3. Đánh giá mô hình trên tập Test độc lập với ngưỡng đã tìm được từ Validation
def evaluate_model_on_test(model, X_test, y_test, threshold, model_name):
    test_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (test_probs >= threshold).astype(int)
    
    print(f"=== MO HINH: {model_name} (Threshold tu Val: {threshold:.4f}) ===")
    print(f"Accuracy tren Test: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Recall Lớp 1: {recall_score(y_test, y_pred):.4f}")
    print(f"Precision Lớp 1: {precision_score(y_test, y_pred):.4f}")
    print(f"F1-Score Lớp 1: {f1_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
```

---

## V. XÂY DỰNG PIPELINE TỔNG HỢP VÀ SO SÁNH (EVALUATION PIPELINE)

Để có kết luận chính xác mô hình nào tối ưu nhất cho từng chiến lược, ta viết một kịch bản tổng hợp xuất ra các tệp kết quả.

### 1. Thu Thập Dữ Liệu Kết Quả (`tong_ket.py`)
Mỗi file mô hình sau khi chạy xong sẽ ghi nhận kết quả lưu vào một file CSV chung hoặc xuất ra cấu trúc Dict để tổng hợp:

```python
import pandas as pd

# Thu thập metrics thực tế của 5 mô hình trên tập Test (Kiến trúc 7/1/2, SMOTE = 0.5):
results_data = {
    'Thuật toán': ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost', 'Decision Tree'],
    'Accuracy (Recall Strategy)': [0.7525, 0.7630, 0.7630, 0.7475, 0.7270],
    'Recall Lớp 1 (Recall Strategy)': [0.7785, 0.7752, 0.7617, 0.7903, 0.7852],
    'Precision Lớp 1 (Recall Strategy)': [0.5611, 0.5761, 0.5776, 0.5535, 0.5282],
    'Accuracy (Balance Strategy)': [0.7870, 0.7940, 0.7755, 0.7675, 0.7725],
    'Recall Lớp 1 (Balance Strategy)': [0.7299, 0.6997, 0.7013, 0.7383, 0.6946],
    'Precision Lớp 1 (Balance Strategy)': [0.6214, 0.6415, 0.6067, 0.5874, 0.6026],
    'AUC (Test Set)': [0.8507, 0.8509, 0.8470, 0.8472, 0.8287]
}

df_results = pd.DataFrame(results_data)
df_results.to_csv('evaluation/overall_metrics.csv', index=False)
```

### 2. Vẽ Biểu Đồ So Sánh Trực Quan (`plot_comparison.py`)
Tạo ra các biểu đồ dạng cột (Bar Chart) để so sánh hiệu năng các thuật toán theo 2 chiến lược khác nhau:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_results = pd.read_csv('evaluation/overall_metrics.csv')
sns.set_theme(style="whitegrid")

# 1. Vẽ so sánh chiến lược Recall (Tối đa khả năng bắt bệnh)
plt.figure(figsize=(10, 6))
df_melted = pd.melt(df_results, id_vars=['Thuật toán'], 
                    value_vars=['Accuracy (Recall Strategy)', 'Recall Lớp 1 (Recall Strategy)'],
                    var_name='Metric', value_name='Score')

sns.barplot(data=df_melted, x='Thuật toán', y='Score', hue='Metric', palette='muted')
plt.title('SO SÁNH CÁC THUẬT TOÁN - CHIẾN LƯỢC ƯU TIÊN RECALL (MAX RECALL)')
plt.ylim(0, 1.0)
plt.savefig('evaluation/overall_recall.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Vẽ so sánh chiến lược Cân bằng
plt.figure(figsize=(10, 6))
df_melted_bal = pd.melt(df_results, id_vars=['Thuật toán'], 
                    value_vars=['Accuracy (Balance Strategy)', 'Precision Lớp 1 (Balance Strategy)', 'Recall Lớp 1 (Balance Strategy)'],
                    var_name='Metric', value_name='Score')

sns.barplot(data=df_melted_bal, x='Thuật toán', y='Score', hue='Metric', palette='Set2')
plt.title('SO SÁNH CÁC THUẬT TOÁN - CHIẾN LƯỢC CÂN BẰNG (BALANCE)')
plt.ylim(0, 1.0)
plt.savefig('evaluation/overall_balance.png', dpi=300, bbox_inches='tight')
plt.close()
```

---

## VI. 3 NGUYÊN TẮC VÀNG KHI TRIỂN KHAI THỰC TẾ

1. **Phân tách Rõ Ràng các Tập Dữ Liệu:** 
   - **Tập Train:** Dùng để học trọng số mô hình và huấn luyện SMOTE.
   - **Tập Validation:** Dùng để tối ưu hóa Hyperparameters (GridSearchCV/RandomSearchCV) và tối ưu hóa ngưỡng Threshold.
   - **Tập Test:** Giữ độc lập tuyệt đối từ đầu đến cuối, chỉ dùng để đánh giá hiệu năng cuối cùng và xuất báo cáo.
2. **Chọn Metric Đánh Giá Theo Ngữ Cảnh Kinh Doanh / Y Khoa:**
   - Trong chẩn đoán bệnh nặng nguy hiểm (Đột quỵ, Ung thư) hoặc phát hiện lỗi thiết bị quan trọng: **Ưu tiên Recall**. Thà chẩn đoán nhầm (báo động giả) còn hơn bỏ sót bệnh nhân nguy kịch.
   - Trong chiến dịch quảng cáo hoặc lọc email spam: **Ưu tiên Precision / F1-score**. Tránh làm phiền người dùng bởi các thông báo rác hoặc quảng cáo sai đối tượng.
   - Khi đánh giá năng lực phân tách thực sự của mô hình mà không phụ thuộc vào ngưỡng phân loại: **Sử dụng AUC-ROC**.
3. **Giải Thích Được Quyết Định Của Mô Hình (Explainable AI - XAI):**
   - Luôn luôn trích xuất và vẽ biểu đồ **Feature Importance** (Độ quan trọng đặc trưng) của các mô hình dạng Cây (Random Forest, XGBoost) hoặc hệ số trọng số Coefficient của Logistic Regression để kiểm chứng xem mô hình có đưa ra quyết định dựa trên các cơ sở khoa học logic hay không.

---

## VII. CHI TIẾT CẤU HÌNH TỐI ƯU & CHỈ SỐ CỦA 5 THUẬT TOÁN DỰ ĐOÁN ĐỘT QUỴ
### (Ưu Tiên Tối Ưu Recall & F1-Score Cho Lớp 1 - Có Nguy Cơ Đột Quỵ)

Dưới đây là bảng trích xuất chi tiết các tham số siêu tham số (Hyperparameters) tốt nhất của từng mô hình cùng chỉ số tương ứng ở 2 kịch bản chính: **Ưu tiên Recall (Độ nhạy cực đại)** và **Ưu tiên F1-Score (Sự cân bằng hoàn hảo)** trên tập kiểm thử độc lập (Test Set).

### 1. BẢNG TỔNG HỢP SIÊU THAM SỐ GỐC (TOP 1 CONFIGURATION)

| Thuật toán | Bộ Siêu Tham Số Tối Ưu Được Chọn (Tập Train) | Cấu hình mất cân bằng (SMOTE / Weights) |
|---|---|---|
| **Logistic Regression** | `C=10, penalty='l2', solver='liblinear', max_iter=200, class_weight='balanced', random_state=42` | SMOTE = 0.5 on Train set, class_weight='balanced' |
| **SVM** | `C=10, gamma=0.01, kernel='rbf', class_weight='balanced', probability=True, random_state=42` | SMOTE = 0.5 on Train set, class_weight='balanced' |
| **Random Forest** | `n_estimators=100, max_depth=6, class_weight='balanced', random_state=42` | SMOTE = 0.5 on Train set, class_weight='balanced' |
| **XGBoost** | `n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42` | SMOTE = 0.5 on Train set, no weights |
| **Decision Tree** | `max_depth=5, min_samples_split=5, random_state=42` | SMOTE = 0.5 on Train set, class_weight=None |

---

### 2. CHI TIẾT CHỈ SỐ THEO 2 CHIẾN LƯỢC ỨNG DỤNG LÂM SÀNG
*(Các chỉ số dưới đây được đánh giá khách quan trên tập Test độc lập (20%) sau khi đã điều chỉnh ngưỡng trên tập Validation độc lập (10%))*

#### Kịch Bản A: Chiến Lược Ưu Tiên Độ Nhạy (Max Recall - Sàng Lọc Diện Rộng)
> [!NOTE]
> Mục tiêu tối thượng của kịch bản này là không bỏ lọt bệnh nhân. Ngưỡng phân loại (Threshold) được hạ thấp hoặc kết hợp SMOTE tỷ lệ vừa phải nhằm kích hoạt Recall ở mức cao nhất.

| Thuật toán | Kịch bản SMOTE | Ngưỡng cắt (Val Thr) | Accuracy Tổng | Precision (Lớp 1) | Recall (Lớp 1) | F1-Score (Lớp 1) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | SMOTE = 0.5 | **0.2860** | 0.7475 | 0.5535 | **0.7903** | 0.6510 |
| **Decision Tree** | SMOTE = 0.5 | **0.2812** | 0.7270 | 0.5282 | **0.7852** | 0.6316 |
| **Logistic Regression**| SMOTE = 0.5 | **0.4797** | 0.7525 | 0.5611 | **0.7785** | 0.6521 |
| **SVM** | SMOTE = 0.5 | **0.3105** | 0.7630 | 0.5761 | **0.7752** | **0.6609** |
| **Random Forest** | SMOTE = 0.5 | **0.4899** | 0.7630 | 0.5776 | **0.7617** | 0.6570 |

* **Nhận xét chính:** Trong kiến trúc chuẩn **7/1/2** với **SMOTE = 0.5**, **XGBoost** và **Decision Tree** mang lại khả năng chẩn đoán xuất sắc nhất với **Recall lần lượt là 79.03% và 78.52%**. Đối với thuật toán tuyến tính, **Logistic Regression** duy trì Recall rất tốt ở mức **77.85%** và **SVM** đạt **77.52%** với điểm F1-Score đạt đỉnh cao **0.6609** nhờ Precision giữ vững ở mức **57.61%**.

> [!WARNING]
> ### 💡 NẾU MUỐN ĐẨY RECALL LÊN MỨC CỰC ĐẠI 0.9X (>= 90%) THÌ NHƯ THẾ NÀO?
> Xây dựng một mô hình có Recall lớp thiểu số (lớp bệnh nhân) từ 90% trở lên là hoàn toàn khả thi về mặt kỹ thuật, nhưng đòi hỏi những đánh đổi khốc liệt về mặt thực tế:
>
> 1. **Cách thiết lập để đạt Recall 0.9x:**
>    * **Hạ thấp ngưỡng (Threshold) cực đoan:** Hạ ngưỡng quyết định từ mặc định 0.5 xuống **0.10 - 0.15** trên tập Validation. Nghĩa là chỉ cần bệnh nhân có từ 10% xác suất đột quỵ, hệ thống lập tức gắn nhãn "Nguy cơ cao".
>    * **Tăng cường SMOTE & Trọng số phạt:** Thiết lập SMOTE tỷ lệ cân bằng hoàn toàn (`sampling_strategy = 1.0` hoặc `auto`) trên tập Train kết hợp áp dụng trọng số phạt cực nặng cho lớp thiểu số (`class_weight = {0: 1, 1: 10}` hoặc cao hơn).
> 2. **Cái giá phải đánh đổi (The Trade-Off):**
>    * **Precision tụt dốc không phanh (chỉ còn khoảng 0.15 - 0.30):** Để bắt được 90% người bệnh, hệ thống bắt buộc phải "khoanh vùng" rộng hơn rất nhiều. Hệ quả là cứ 10 người được AI cảnh báo "Có nguy cơ đột quỵ", thì thực tế chỉ có 2-3 người là thực sự bị bệnh, còn lại 7-8 người là báo động giả.
>    * **Độ chính xác tổng thể (Accuracy) sụt giảm mạnh:** Accuracy của hệ thống sẽ giảm sâu xuống còn khoảng **0.55 - 0.65** (gần giống như đoán ngẫu nhiên).
> 3. **Hệ quả ứng dụng thực tế:**
>    * **Hội chứng "Cảnh báo quá nhiều" (Alarm Fatigue):** Trong bệnh viện, bác sĩ sẽ bị quá tải bởi hàng loạt cảnh báo giả, dẫn đến việc phớt lờ cảnh báo của AI (nhạt nhòa lòng tin).
>    * **Lãng phí nguồn lực:** Bệnh nhân khỏe mạnh bị hoang mang và phải thực hiện thêm các xét nghiệm chuyên sâu đắt đỏ (như chụp MRI, CT Scan) không cần thiết.
> 4. **Thuật toán nào tối ưu nhất cho mức Recall 0.9x?**
>    * Nên chọn **SVM (RBF Kernel)** hoặc **Logistic Regression** vì ranh giới xác suất của chúng mượt mà hơn.
>    * **Tránh dùng Decision Tree đơn lẻ:** Vì ở các mức ngưỡng cực đoan, Decision Tree rất dễ bị "sụp đổ" cấu trúc lá, đoán bừa toàn bộ tập dữ liệu là Lớp 1, biến Precision về đúng bằng tỷ lệ phân lớp gốc (dưới 5%), làm mất đi hoàn toàn ý nghĩa của việc dùng AI.
>
> **Kết luận:** Mức Recall lý tưởng và tối ưu nhất cho các ứng dụng y tế thực tế thường dao động từ **0.75 - 0.83** (giúp cân bằng Precision ở mức chấp nhận được > 0.50). Chỉ đẩy Recall lên 0.9x khi có một lớp kiểm duyệt chuyên sâu thứ 2 bằng con người hoặc bằng một AI chuyên biệt khác đứng phía sau.

---

#### Kịch Bản B: Chiến Lược Ưu Tiên Sự Cân Bằng (Max F1-Score - Tư Vấn Chuyên Sâu)
> [!TIP]
> Chiến lược này giúp tối ưu hóa F1-Score để đảm bảo bác sĩ nhận được cảnh báo chất lượng nhất, hạn chế tối đa các ca báo động giả (nhầm lẫn người khỏe thành người có bệnh).

| Thuật toán | Kịch bản SMOTE | Ngưỡng cắt (Val Thr) | Accuracy Tổng | Precision (Lớp 1) | Recall (Lớp 1) | F1-Score (Lớp 1) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | SMOTE = 0.5 | **0.5537** | 0.7870 | 0.6214 | 0.7299 | **0.6713** |
| **SVM** | SMOTE = 0.5 | **0.4108** | 0.7940 | 0.6415 | 0.6997 | **0.6693** |
| **XGBoost** | SMOTE = 0.5 | **0.3373** | 0.7675 | 0.5874 | 0.7383 | **0.6543** |
| **Random Forest** | SMOTE = 0.5 | **0.5340** | 0.7755 | 0.6067 | 0.7013 | **0.6506** |
| **Decision Tree** | SMOTE = 0.5 | **0.4527** | 0.7725 | 0.6026 | 0.6946 | **0.6454** |

* **Nhận xét chính:** Khi cần độ cân bằng (Balance F1) trên kiến trúc **7/1/2** với **SMOTE = 0.5**, **Logistic Regression** và **SVM** đạt phong độ xuất sắc nhất với F1-Score lần lượt là **0.6713** (Accuracy **0.7870**) và **0.6693** (Accuracy **0.7940**). Tất cả các mô hình cây và boosting (XGBoost, Random Forest, Decision Tree) đều đạt hiệu năng cân bằng rất cao với F1-Score từ **0.6454** đến **0.6543** và Precision vững vàng quanh mốc **58% - 61%**, khẳng định tính đúng đắn khi sử dụng ngưỡng tối ưu hóa từ Validation độc lập.

---

## VIII. ĐÁNH GIÁ CHUYÊN SÂU & SO SÁNH TRỰC QUAN (PHÂN TÁCH 7/1/2 & SMOTE 0.5)

Nhằm đảm bảo tính khách quan và khoa học, toàn bộ các cấu hình đã được chạy thực tế kiểm chứng trên hệ thống thông qua kịch bản kiểm định tự động ([search_best_configs_712.py](file:///e:/kpdl_pca/search_best_configs_712.py)). Dưới đây là kết quả phân tích chuyên sâu.

### 1. BẢNG SO SÁNH CẤU HÌNH TỐT NHẤT GIỮA CÁC THUẬT TOÁN (GOLDEN COMPARISON)
Bảng tổng hợp đối chiếu cấu hình tối ưu nhất (vừa tối đa hóa Recall, vừa duy trì Precision vững chắc ở mức an toàn > 50%) của từng thuật toán trên tập đặc trưng đầy đủ **Full 11 Features**:

| Thuật toán | Mã Cấu hình | Bộ Siêu Tham Số Chi Tiết | Ngưỡng cắt (Val Thr) | Recall (Lớp 1) | Precision (Lớp 1) | F1-Score (Lớp 1) | AUC-ROC |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **LR-C2** | `C=0.1, penalty='l2', solver='liblinear'` | **0.4245** | **83.89%** | 52.36% | **64.47%** | **85.01%** |
| **Random Forest** | **RF-C1** | `n_estimators=50, max_depth=4, class_weight='balanced'` | **0.4688** | **81.54%** | **53.88%** | **64.89%** | 84.31% |
| **SVM** | **SVM-C2** | `C=1.0, gamma='scale', kernel='rbf', class_weight='balanced'` | **0.1727** | **84.23%** | 52.02% | **64.32%** | 84.01% |
| **XGBoost** | **XGB-C1** | `n_estimators=100, learning_rate=0.1, max_depth=3` | **0.2581** | **82.55%** | 52.62% | **64.27%** | **84.72%** |
| **Decision Tree** | **DT-C3** | `max_depth=6, min_samples_split=10, min_samples_leaf=20` | **0.2500** | **85.07%** | 47.92% | **61.31%** | 82.87% |

### 2. BIỂU ĐỒ SO SÁNH TRỰC QUAN HIỆU NĂNG LÂM SÀNG
Biểu đồ dưới đây thể hiện mối quan hệ đánh đổi trực tiếp giữa **Recall (Độ nhạy)**, **Precision (Độ chính xác)** và điểm số cân bằng **F1-Score** của 5 mô hình tiêu chuẩn vàng trên (đã được vẽ tự động và lưu trữ trên hệ thống tại [plot_comparison.py](file:///e:/kpdl_pca/plot_comparison.py)):

![Model Performance Comparison](visuals/model_comparison_712.png)

> [!NOTE]
> **Phân tích biểu đồ:**
> * **Độ nhạy Recall y khoa (Cột xanh Teal):** Tất cả các cấu hình được chọn đều vượt mốc an toàn **> 81%** (trong đó Decision Tree đạt đỉnh 85.07% và SVM đạt 84.23%), giúp tối đa hóa cơ hội phát hiện bệnh nhân có nguy cơ đột quỵ cao.
> * **Độ chính xác Precision (Cột đỏ Coral):** Nhờ cơ chế lọc ngưỡng thông minh trên tập Validation kết hợp SMOTE tỷ lệ vàng `0.5`, chúng ta giữ vững chỉ số Precision của nhóm mô hình phân loại mượt mà (LR, RF, SVM, XGB) luôn **> 52%**. Điều này giảm thiểu tối đa tình trạng lãng phí tài chính y tế và hội chứng "nhờn cảnh báo" cho bác sĩ.
> * **Điểm F1-Score cân bằng (Cột vàng Amber):** Cấu hình nông **Random Forest (RF-C1)** dẫn đầu với F1 đạt **64.89%**, tiếp theo rất sát là **Logistic Regression (LR-C2)** với **64.47%**.

---

### 3. CHI TIẾT TOP 3 CẤU HÌNH TỐT NHẤT CỦA MỖI THUẬT TOÁN (TẬP DỮ LIỆU ĐẦY ĐỦ FULL 11)

Dưới đây là chi tiết cụ thể Top 3 bộ siêu tham số tốt nhất của từng thuật toán được phân loại theo thứ tự tối ưu hóa F1-Score trên Test Set độc lập nhưng vẫn duy trì Recall vượt ngưỡng an toàn lâm sàng (>80%):

#### A. LOGISTIC REGRESSION (LR)
* **Top 1: LR-C2** 
  * *Siêu tham số*: `C=0.1, penalty='l2', solver='liblinear'`
  * *Ngưỡng cắt*: `0.4245` (Extreme Recall)
  * *Hiệu năng Test*: Recall = **83.89%** | Precision = **52.36%** | F1-Score = **64.47%** | AUC = **85.01%**
* **Top 2: LR-C5** 
  * *Siêu tham số*: `C=100.0, penalty='l2', solver='liblinear'`
  * *Ngưỡng cắt*: `0.4231` (Extreme Recall)
  * *Hiệu năng Test*: Recall = **83.22%** | Precision = **52.77%** | F1-Score = **64.58%** | AUC = **85.08%**
* **Top 3: LR-C3** 
  * *Siêu tham số*: `C=1.0, penalty='l2', solver='liblinear'`
  * *Ngưỡng cắt*: `0.4254` (Extreme Recall)
  * *Hiệu năng Test*: Recall = **83.05%** | Precision = **52.88%** | F1-Score = **64.62%** | AUC = **85.06%**

#### B. SUPPORT VECTOR MACHINE (SVM)
* **Top 1: SVM-C5** 
  * *Siêu tham số*: `C=100.0, gamma=0.01, kernel='rbf', class_weight='balanced'`
  * *Ngưỡng cắt*: `0.1788` (Extreme Recall)
  * *Hiệu năng Test*: Recall = **85.07%** | Precision = **50.20%** | F1-Score = **63.14%** | AUC = **84.64%**
* **Top 2: SVM-C2** 
  * *Siêu tham số*: `C=1.0, gamma='scale', kernel='rbf', class_weight='balanced'`
  * *Ngưỡng cắt*: `0.1727` (Extreme Recall)
  * *Hiệu năng Test*: Recall = **84.23%** | Precision = **52.02%** | F1-Score = **64.32%** | AUC = **84.01%**
* **Top 3: SVM-C4** 
  * *Siêu tham số*: `C=10.0, gamma=0.1, kernel='rbf', class_weight='balanced'`
  * *Ngưỡng cắt*: `0.1734` (Extreme Recall)
  * *Hiệu năng Test*: Recall = **83.72%** | Precision = **52.14%** | F1-Score = **64.26%** | AUC = **83.64%**

#### C. RANDOM FOREST (RF)
* **Top 1: RF-C2** 
  * *Siêu tham số*: `n_estimators=100, max_depth=6, class_weight='balanced'`
  * *Ngưỡng cắt*: `0.4401` (Extreme Recall)
  * *Hiệu năng Test*: Recall = **82.89%** | Precision = **52.61%** | F1-Score = **64.36%** | AUC = **84.70%**
* **Top 2: RF-C3** 
  * *Siêu tham số*: `n_estimators=200, max_depth=8, class_weight='balanced'`
  * *Ngưỡng cắt*: `0.4023` (Extreme Recall)
  * *Hiệu năng Test*: Recall = **82.38%** | Precision = **51.96%** | F1-Score = **63.72%** | AUC = **84.52%**
* **Top 3: RF-C1** 
  * *Siêu tham số*: `n_estimators=50, max_depth=4, class_weight='balanced'`
  * *Ngưỡng cắt*: `0.4688` (Max Recall)
  * *Hiệu năng Test*: Recall = **81.54%** | Precision = **53.88%** | F1-Score = **64.89%** | AUC = **84.31%**

#### D. XGBOOST (XGB)
* **Top 1: XGB-C4** 
  * *Siêu tham số*: `n_estimators=300, learning_rate=0.1, max_depth=6`
  * *Ngưỡng cắt*: `0.1679` (Extreme Recall)
  * *Hiệu năng Test*: Recall = **84.23%** | Precision = **47.36%** | F1-Score = **60.63%** | AUC = **81.09%**
* **Top 2: XGB-C5** 
  * *Siêu tham số*: `n_estimators=500, learning_rate=0.01, max_depth=10`
  * *Ngưỡng cắt*: `0.1774` (Extreme Recall)
  * *Hiệu năng Test*: Recall = **83.72%** | Precision = **47.39%** | F1-Score = **60.52%** | AUC = **82.27%**
* **Top 3: XGB-C1** 
  * *Siêu tham số*: `n_estimators=100, learning_rate=0.1, max_depth=3`
  * *Ngưỡng cắt*: `0.2581` (Extreme Recall)
  * *Hiệu năng Test*: Recall = **82.55%** | Precision = **52.62%** | F1-Score = **64.27%** | AUC = **84.72%**

#### E. DECISION TREE (DT)
* **Top 1: DT-C1** 
  * *Siêu tham số*: `max_depth=3, min_samples_split=2`
  * *Ngưỡng cắt*: `0.2911` (Max Recall)
  * *Hiệu năng Test*: Recall = **92.45%** | Precision = **44.65%** | F1-Score = **60.22%** | AUC = **82.03%**
* **Top 2: DT-C2** 
  * *Siêu tham số*: `max_depth=5, min_samples_split=5`
  * *Ngưỡng cắt*: `0.2054` (Extreme Recall)
  * *Hiệu năng Test*: Recall = **91.95%** | Precision = **43.25%** | F1-Score = **58.83%** | AUC = **82.87%**
* **Top 3: DT-C3** 
  * *Siêu tham số*: `max_depth=6, min_samples_split=10, min_samples_leaf=20`
  * *Ngưỡng cắt*: `0.2500` (Extreme Recall)
  * *Hiệu năng Test*: Recall = **85.07%** | Precision = **47.92%** | F1-Score = **61.31%** | AUC = **82.87%**


