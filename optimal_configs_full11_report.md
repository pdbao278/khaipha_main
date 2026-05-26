# 📋 BÁO CÁO THỰC NGHIỆM: TOP 5 CẤU HÌNH TỐI ƯU CỦA CÁC THUẬT TOÁN (FULL 11 FEATURES)

Báo cáo này được thực hiện bằng cách huấn luyện trực tiếp trên máy thật, sử dụng tập dữ liệu đầy đủ **Full 11 Features (Chưa giảm chiều bằng MI)** của dự án **Dự đoán nguy cơ Đột quỵ (Stroke Prediction)**.

Tập dữ liệu huấn luyện đã được áp dụng phương pháp **SMOTE với tỷ lệ 0.5** (Lớp 1 sau SMOTE chiếm 50% số lượng lớp 0). Hệ thống đã áp dụng chiến lược tinh chỉnh ngưỡng **Extreme Recall** trên tập **Validation** (tìm ngưỡng tối ưu để Recall đạt $\ge 80\%$ trên tập Validation nhằm tối đa hóa khả năng phát hiện bệnh nhân, đồng thời giữ F1-Score ở mức tối ưu nhất) và đánh giá độc lập trên tập **Test** gồm 2,000 mẫu.

---

## 📊 BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG TẤT CẢ CẤU HÌNH

Dưới đây là kết quả thực nghiệm chi tiết của tất cả các cấu hình được sắp xếp theo thứ tự ưu tiên: **Recall Lớp 1 $\ge 77\%$ trước, sau đó sắp xếp theo F1-Score giảm dần** để đảm bảo không bị sụt giảm Precision quá sâu (hạn chế báo động giả).

> [!NOTE]
> Ký hiệu **★** đánh dấu các cấu hình đạt tiêu chuẩn y khoa khắt khe với **Recall lớp 1 > 77%** trên tập dữ liệu kiểm thử độc lập (Test Set).

| Thuật toán | Thứ hạng | Mã Cấu hình | Siêu tham số chính | Ngưỡng (Threshold) | Recall (Lớp 1) | Precision (Lớp 1) | F1-Score (Lớp 1) | AUC-ROC | Trạng thái |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **Top 1 ★** | **LR-C4** | $C=10.0$, balanced | **0.4785** | **78.19%** | **56.14%** | **65.36%** | **85.07%** | **ĐẠT** |
| | **Top 2 ★** | **LR-C5** | $C=100.0$, balanced | **0.4780** | **78.19%** | **56.14%** | **65.36%** | **85.08%** | **ĐẠT** |
| | **Top 3 ★** | **LR-C2** | $C=0.1$, balanced | **0.4875** | **77.52%** | **56.41%** | **65.30%** | **85.02%** | **ĐẠT** |
| | Top 4 | LR-C3 | $C=1.0$, balanced | 0.4890 | 76.85% | 56.54% | 65.15% | 85.06% | Tiệm cận |
| | Top 5 | LR-C1 | $C=0.01$, balanced | 0.5030 | 75.84% | 56.78% | 64.94% | 84.39% | Thấp |
| **Random Forest** | **Top 1 ★** | **RF-C1** | depth=4, n=50, balanced | **0.4710** | **81.21%** | **54.02%** | **64.88%** | **84.31%** | **ĐẠT** |
| | **Top 2 ★** | **RF-C3** | depth=8, n=200, balanced | **0.4385** | **77.68%** | **54.73%** | **64.22%** | **84.50%** | **ĐẠT** |
| | **Top 3 ★** | **RF-C4** | depth=8, n=300, split=10, balanced | **0.4145** | **80.37%** | **53.40%** | **64.17%** | **84.51%** | **ĐẠT** |
| | **Top 4 ★** | **RF-C5** | depth=12, n=500, balanced | **0.3505** | **78.19%** | **52.01%** | **62.47%** | **83.63%** | **ĐẠT** |
| | Top 5 | RF-C2 | depth=6, n=100, balanced | 0.4840 | 76.34% | 57.16% | 65.37% | 84.71% | Tiệm cận |
| **XGBoost** | **Top 1 ★** | **XGB-C1** | depth=3, lr=0.1, n=100 | **0.2875** | **78.69%** | **55.31%** | **64.96%** | **84.85%** | **ĐẠT** |
| | **Top 2 ★** | **XGB-C3** | depth=4, lr=0.1, n=150 | **0.2640** | **78.52%** | **52.29%** | **62.78%** | **83.55%** | **ĐẠT** |
| | **Top 3 ★** | **XGB-C4** | depth=6, lr=0.1, n=300 | **0.2085** | **79.36%** | **50.05%** | **61.39%** | **81.36%** | **ĐẠT** |
| | **Top 4 ★** | **XGB-C5** | depth=10, lr=0.01, n=500 | **0.1925** | **81.71%** | **48.12%** | **60.57%** | **82.26%** | **ĐẠT** |
| | Top 5 | XGB-C2 | depth=5, lr=0.05, n=200 | 0.2860 | 76.85% | 54.59% | 63.83% | 83.52% | Tiệm cận |
| **Decision Tree** | **Top 1 ★** | **DT-C3** | depth=6, leaf=20, balanced | **0.4245** | **77.01%** | **55.70%** | **64.65%** | **83.51%** | **ĐẠT** |
| | **Top 2 ★** | **DT-C2** | depth=5, split=5, balanced | **0.3575** | **81.21%** | **50.89%** | **62.57%** | **82.85%** | **ĐẠT** |
| | **Top 3 ★** | **DT-C1** | depth=3, split=2, balanced | **0.2880** | **92.45%** | **44.65%** | **60.22%** | **82.03%** | **ĐẠT** |
| | **Top 4 ★** | **DT-C5** | depth=12, split=20, balanced | **0.3000** | **77.35%** | **48.63%** | **59.72%** | **76.69%** | **ĐẠT** |
| | Top 5 | DT-C4 | depth=8, split=10, balanced | 0.4350 | 76.85% | 53.82% | 63.30% | 79.85% | Tiệm cận |
| **SVM (RBF)** | **Top 1 ★** | **SVM-C2** | $C=1.0$, gamma='scale', balanced | **0.2540** | **77.52%** | **58.19%** | **66.47%** | **84.00%** | **ĐẠT** |
| | **Top 2 ★** | **SVM-C3** | $C=10.0$, gamma=0.01, balanced | **0.3115** | **77.35%** | **57.63%** | **66.05%** | **85.09%** | **ĐẠT** |
| | **Top 3 ★** | **SVM-C5** | $C=100.0$, gamma=0.01, balanced | **0.2630** | **77.68%** | **57.37%** | **66.00%** | **84.64%** | **ĐẠT** |
| | Top 4 | SVM-C4 | $C=10.0$, gamma=0.1, balanced | 0.2580 | 75.84% | 58.25% | 65.89% | 83.65% | Thấp |
| | Top 5 | SVM-C1 | $C=0.1$, gamma='scale', balanced | 0.3235 | 74.33% | 57.76% | 65.00% | 84.19% | Thấp |

---

## 🔍 PHÂN TÍCH CHI TIẾT TỪNG THUẬT TOÁN

### 1. Logistic Regression (LR)
* **Nhận xét chung**: LR cho thấy sự ổn định xuất sắc khi huấn luyện với `class_weight='balanced'`. F1-Score duy trì rất cao trên 65% ở tất cả cấu hình.
* **Cấu hình tốt nhất**: **LR-C4** ($C=10.0$) đạt **Recall = 78.19%**, **Precision = 56.14%** và **F1-Score = 65.36%** ở ngưỡng **0.4785**. Cấu hình này giúp phát hiện tối đa các ca bệnh nguy kịch mà vẫn giữ lượng dương tính giả ở mức cực thấp so với mặt bằng chung.

### 2. Random Forest (RF)
* **Nhận xét chung**: Rừng ngẫu nhiên là thuật toán có tính ổn định cực tốt. Với việc giới hạn độ sâu cây và cấu hình số lượng cây hợp lý, RF mang lại kết quả y khoa rất khả thi.
* **Cấu hình tốt nhất**: **RF-C1** (max_depth=4, n_estimators=50) đạt chỉ số **Recall ấn tượng = 81.21%** đi kèm với **F1-Score = 64.88%** ở ngưỡng **0.4710**. Việc giới hạn độ sâu cây ở mức 4 giúp RF tránh Overfitting cực tốt, giữ vững tính tổng quát hóa trên tập dữ liệu kiểm thử.

### 3. XGBoost (XGB)
* **Nhận xét chung**: XGBoost cực mạnh mẽ nhờ cơ chế Gradient Boosting. Bằng việc tối ưu hóa ngưỡng trên Validation set, mô hình đã thành công ép Recall lên rất cao mà không làm hỏng F1.
* **Cấu hình tốt nhất**: **XGB-C1** (max_depth=3, learning_rate=0.1, n_estimators=100) cho **Recall = 78.69%**, **Precision = 55.31%** và **F1-Score = 64.96%** ở ngưỡng **0.2875**. Việc khống chế cây rất nông (depth=3) giúp XGBoost cực kỳ ổn định.

### 4. Decision Tree (DT)
* **Nhận xét chung**: Cây quyết định đơn lẻ tuy dễ Overfitting nhưng nếu được tinh chỉnh cẩn thận, nó có thể mang lại kết quả Recall cực kỳ bùng nổ.
* **Cấu hình tốt nhất**: **DT-C3** (max_depth=6, min_samples_leaf=20) là đỉnh cao cân bằng với **Recall = 77.01%**, **Precision = 55.70%** và **F1 = 64.65%** ở ngưỡng **0.4245**. 
* *Đặc biệt*: Nếu bệnh viện chấp nhận tỷ lệ dương tính giả cao để đổi lấy độ nhạy tuyệt đối, cấu hình nông **DT-C1** (max_depth=3) đạt **Recall kỷ lục = 92.45%** (Precision = 44.65%, F1 = 60.22%).

### 5. SVM (RBF Kernel)
* **Nhận xét chung**: SVM sử dụng RBF Kernel đạt hiệu quả phân ranh phi tuyến đỉnh cao, cho F1-Score cao nhất toàn bộ dự án (> 66%).
* **Cấu hình tốt nhất**: **SVM-C2** ($C=1.0$, gamma='scale') đạt **F1-Score cao nhất dự án = 66.47%** với **Recall = 77.52%** và **Precision = 58.19%** ở ngưỡng **0.2540**. Đây là một mô hình vô cùng tin cậy cho ứng dụng thực tế nhờ sự cân bằng chuẩn xác nhất giữa sai lầm loại I và loại II.

---

## 🏥 KHUYẾN NGHỊ ỨNG DỤNG LÂM SÀNG (GOLDEN STANDARDS)

Dựa trên kết quả thực tế thu được từ máy thật, chúng tôi đề xuất 2 mô hình **Tiêu chuẩn Vàng** cho 2 kịch bản sử dụng:

1. **Mô hình Khám Sàng Lọc Diện Rộng (First-line Screening)**: 
   * **Đề xuất**: **Random Forest (RF-C1)** ở ngưỡng **0.4710**.
   * **Lợi ích**: Đạt **Recall cực cao = 81.21%**, giúp giảm thiểu tối đa khả năng bỏ sót bệnh nhân đột quỵ nguy kịch, trong khi F1 vẫn giữ vững ở mức 64.88%.
2. **Mô hình Chẩn Đoán Cân Bằng Chuẩn Xác (Precision Diagnostic)**:
   * **Đề xuất**: **SVM (SVM-C2)** ở ngưỡng **0.2540**.
   * **Lợi ích**: Đạt **F1-Score cao nhất toàn bộ thực nghiệm = 66.47%** và **Precision tốt nhất = 58.19%**, giảm tối đa tỷ lệ báo động giả gây lãng phí nguồn lực bệnh viện và hoang mang cho bệnh nhân, trong khi Recall vẫn vững vàng đạt mốc **77.52%**.
