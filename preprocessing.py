import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# --- Bước 1: Khởi Tạo Dự Án & Đọc Dữ Liệu ---
# Tạo thư mục lưu ảnh nếu chưa có
os.makedirs('visuals', exist_ok=True)

# Đọc dữ liệu đột quỵ
df = pd.read_csv('stroke_data.csv')
print(f"Kich thuoc tep du lieu: {df.shape}")
print(f"Cac cot thuoc tinh: {df.columns.tolist()}")

# Vẽ phân phối nhãn mục tiêu (Stroke)
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Stroke', palette='viridis')
plt.title('Phân phối Nhãn Mục Tiêu (Stroke)')
plt.savefig('visuals/1_distribution.png', bbox_inches='tight')
plt.close()

# --- Bước 3: Xử Lý Ngoại Lệ Bằng Phương Pháp IQR (Interquartile Range) ---
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

# --- Bước 4: Mã Hóa Biến Phân Loại (Categorical Encoding) ---
# 1. Label Mapping cho cột Gender (chỉ có 2 nhóm)
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

# 2. Ordinal Encoding cho cột SES bằng cách định nghĩa Map cụ thể
ses_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['SES'] = df['SES'].map(ses_map)

# 3. One-Hot Encoding cho cột Smoking_Status (nhiều nhóm không thứ tự)
df = pd.get_dummies(df, columns=['Smoking_Status'], prefix='Smoking_Status', dtype=int)

# --- Bước 5: Chuẩn Hóa Đặc Trưng (Feature Scaling) ---
scaler = StandardScaler()
df_scaled = df.copy()

# Chuẩn hóa các thuộc tính liên tục
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Lưu lại tập dữ liệu đã sạch sẽ hoàn toàn
df_scaled.to_csv('data_processed.csv', index=False)
print("Tien xu ly hoan tat! File da luu vao: 'data_processed.csv'")
