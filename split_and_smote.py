import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# 1. Đọc dữ liệu đã tiền xử lý
df = pd.read_csv('data_processed.csv')

# Chia các thuộc tính (X) và nhãn (y)
X = df.drop('Stroke', axis=1)
y = df['Stroke']

# 2. Tách 20% dữ liệu làm tập Test độc lập
X_temp_train, X_test, y_temp_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Tách phần còn lại (80%) thành tập Train (70% tổng) và Validation (10% tổng)
# 10% tổng số mẫu tương ứng với 10/80 = 12.5% của tập temp_train
X_train, X_val, y_train, y_val = train_test_split(
    X_temp_train, y_temp_train, test_size=0.125, random_state=42, stratify=y_temp_train
)

# 4. Chỉ áp dụng SMOTE trên tập huấn luyện X_train để chống rò rỉ dữ liệu (Data Leakage)
# Tỷ lệ sampling_strategy=0.5 nghĩa là Lớp 1 sau SMOTE sẽ bằng 50% số lượng của Lớp 0
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 5. In thống kê kích thước các tập
print(f"Kich thuoc tap Train (70%): {X_train.shape[0]} mau")
print(f"Kich thuoc tap Val (10%): {X_val.shape[0]} mau")
print(f"Kich thuoc tap Test (20%): {X_test.shape[0]} mau")
print(f"Truoc SMOTE (Lop Train): {y_train.value_counts().to_dict()}")
print(f"Sau SMOTE (Lop Train): {y_train_resampled.value_counts().to_dict()}")

# 6. Gộp X và y lại để lưu thành tệp CSV đầy đủ cho từng tập dữ liệu
train_resampled_df = pd.concat([X_train_resampled, y_train_resampled], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Lưu các tệp CSV kết quả
train_resampled_df.to_csv('train_resampled.csv', index=False)
val_df.to_csv('val.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("\nDa luu thanh cong cac tap tin du lieu sau chia tach:")
print("- train_resampled.csv (Tap Train da duoc SMOTE = 0.5)")
print("- val.csv (Tap Validation goc)")
print("- test.csv (Tap Test goc)")
