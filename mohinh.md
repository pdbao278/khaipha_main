# 📋 BÁO CÁO CẤU HÌNH TỐI ƯU CHO PHÂN CHIA 7/1/2 (TOP 3 CẤU HÌNH THEO TỪNG MÔ HÌNH)

Dự án này tập trung vào bài toán **Dự đoán nguy cơ Đột quỵ (Stroke Prediction)** sử dụng tập dữ liệu gồm **10,000 bệnh nhân** với phân tách dữ liệu theo tỷ lệ **70% Train, 10% Validation, 20% Test (7/1/2)**. Tập dữ liệu bị mất cân bằng lớp cao (70% không bị đột quỵ, 30% bị đột quỵ). Do tính chất y khoa, chỉ số **Recall (Độ nhạy) được ưu tiên tối đa** để không bỏ lọt bệnh nhân nguy kịch, nhưng vẫn cần kiểm soát **Precision và F1-Score** ở mức tốt nhất có thể nhằm hạn chế báo động giả gây lãng phí nguồn lực và hoang mang.

Tài liệu này tổng hợp **Top 3 cấu hình tốt nhất** của mỗi thuật toán (thỏa mãn chỉ số **Recall Lớp 1 > 80%** hoặc tiệm cận gần nhất, đồng thời F1-Score không bị sụt giảm quá sâu) trên hai biến thể dữ liệu:
1. **Full 11 Features**: Toàn bộ 11 thuộc tính sau khi tiền xử lý và mã hóa.
2. **MI Features (5 Features)**: Rút gọn còn 5 thuộc tính lâm sàng quan trọng nhất được chọn qua Mutual Information gồm: `Age`, `Hypertension`, `Heart_Disease`, `Avg_Glucose`, `SES`.

---

## 📊 BẢNG SO SÁNH TỔNG QUAN HIỆU NĂNG TỐT NHẤT CỦA CÁC MÔ HÌNH
Dưới đây là cấu hình tốt nhất đạt sự cân bằng tối ưu giữa Recall > 80% và F1-Score của từng thuật toán:

| Thuật toán | Tập dữ liệu | Cấu hình Siêu tham số | Ngưỡng (Threshold) | Recall (Lớp 1) | Precision (Lớp 1) | F1-Score (Lớp 1) | AUC-ROC |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **Full 11** | LR-C2 (C=0.1, Balanced) | **0.4245** | **83.89%** | 52.36% | **64.47%** | **85.01%** |
| **Random Forest** | **Full 11** | RF-C1 (max_depth=4, Balanced) | **0.4688** | **81.54%** | **53.88%** | **64.89%** | 84.31% |
| **SVM** | **Full 11** | SVM-C2 (C=1.0, Balanced) | **0.1727** | **84.23%** | 52.02% | **64.32%** | 84.01% |
| **XGBoost** | **Full 11** | XGB-C1 (max_depth=3) | **0.2581** | **82.55%** | 52.62% | **64.27%** | 84.72% |
| **Decision Tree** | **Full 11** | DT-C3 (depth=6, leaf=20) | **0.2500** | **85.07%** | 47.92% | **61.31%** | 82.87% |

### 🔄 1. BẢNG ĐỐI CHIẾU SỰ TĂNG GIẢM HIỆU NĂNG: FULL 11 FEATURES VS MI 5 FEATURES
Bảng này đối chiếu chi tiết sự thay đổi chỉ số hiệu năng lâm sàng khi thu gọn từ **Tập đầy đủ 11 đặc trưng** xuống **Tập rút gọn 5 đặc trưng Mutual Information (MI)** của từng mô hình tối ưu:

| Thuật toán & Cấu hình so sánh | Chỉ số | Tập Full 11 Features | Tập MI 5 Features | Chênh lệch ($\Delta$ MI vs Full) | Đánh giá xu hướng & Trạng thái |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression** <br> *(LR-C2 vs LR-C2)* | Recall <br> Precision <br> F1-Score <br> AUC-ROC | 83.89% <br> 52.36% <br> 64.47% <br> 85.01% | 82.38% <br> 49.10% <br> 61.53% <br> 81.50% | 🔻 -1.51% <br> 🔻 -3.26% <br> 🔻 -2.94% <br> 🔻 -3.51% | **Giảm nhẹ**: F1-Score và Recall được bảo toàn cực kỳ tốt. Mô hình tuyến tính giữ độ ổn định rất cao khi thu gọn thông tin. |
| **Random Forest** <br> *(RF-C1 vs RF-C4)* | Recall <br> Precision <br> F1-Score <br> AUC-ROC | 81.54% <br> 53.88% <br> 64.89% <br> 84.31% | 81.04% <br> 49.24% <br> 61.26% <br> 80.39% | 🔻 -0.50% <br> 🔻 -4.64% <br> 🔻 -3.63% <br> 🔻 -3.92% | **Cực kỳ ổn định**: Chỉ số Recall y tế quan trọng nhất hầu như được giữ nguyên (chỉ giảm 0.50%). Rất tối ưu chi phí sàng lọc. |
| **SVM** <br> *(SVM-C2 vs SVM-C3)* | Recall <br> Precision <br> F1-Score <br> AUC-ROC | 84.23% <br> 52.02% <br> 64.32% <br> 84.01% | 80.87% <br> 50.16% <br> 61.91% <br> 81.05% | 🔻 -3.36% <br> 🔻 -1.86% <br> 🔻 -2.41% <br> 🔻 -2.96% | **Giảm đều**: Điểm F1-Score vẫn duy trì ở mức cao (>61.9%), Recall vẫn vững vàng vượt mốc an toàn y khoa 80%. |
| **XGBoost** <br> *(XGB-C1 vs XGB-C5)* | Recall <br> Precision <br> F1-Score <br> AUC-ROC | 82.55% <br> 52.62% <br> 64.27% <br> 84.72% | 83.22% <br> 45.38% <br> 58.73% <br> 77.72% | 🔺 +0.67% <br> 🔻 -7.24% <br> 🔻 -5.54% <br> 🔻 -7.00% | **Đánh đổi cao**: Recall tăng nhẹ (+0.67%) nhờ cấu hình cây sâu hơn trên tập MI, nhưng phải đánh đổi bằng việc sụt giảm Precision và F1. |
| **Decision Tree** <br> *(DT-C3 vs DT-C4)* | Recall <br> Precision <br> F1-Score <br> AUC-ROC | 85.07% <br> 47.92% <br> 61.31% <br> 82.87% | 85.07% <br> 44.63% <br> 58.55% <br> 76.74% | 0.00% <br> 🔻 -3.29% <br> 🔻 -2.76% <br> 🔻 -6.13% | **Độ nhạy tuyệt đối**: Bảo toàn chính xác Recall 85.07%, tuy nhiên khả năng phân tách tổng thể (AUC-ROC) bị sụt giảm hơn 6%. |

### 🏥 2. BẢNG SO SÁNH TRỰC DIỆN HAI MÔ HÌNH LÂM SÀNG HÀNG ĐẦU (GOLDEN STANDARDS)
Dưới đây là bảng so sánh đối chiếu sâu hai mô hình được chọn làm **Tiêu chuẩn Vàng (Golden Standards)** cho triển khai thực tế trên tập dữ liệu đầy đủ **Full 11 Features**:

| Chỉ số so sánh | Logistic Regression (LR-C2) | Random Forest (RF-C1) | Chênh lệch (RF-C1 vs LR-C2) | Đánh giá & Định hướng Ứng dụng |
| :--- | :---: | :---: | :---: | :--- |
| **Recall (Độ nhạy Lớp 1)** | **83.89%** | 81.54% | 🔻 -2.35% | **LR-C2 an toàn hơn**: Đột phá hơn trong việc phát hiện tối đa các ca đột quỵ thực tế, giảm tối đa nguy cơ bỏ sót bệnh nhân. |
| **Precision (Độ chính xác Lớp 1)** | 52.36% | **53.88%** | 🔺 +1.52% | **RF-C1 tiết kiệm hơn**: Cung cấp cảnh báo chuẩn xác cao hơn, hạn chế số ca dương tính giả gây lo lắng và lãng phí viện phí. |
| **F1-Score (Cân bằng Lớp 1)** | 64.47% | **64.89%** | 🔺 +0.42% | **RF-C1 hài hòa hơn**: Nhỉnh hơn nhẹ về độ cân bằng tổng thể giữa sai số loại I và loại II. |
| **AUC-ROC (Khả năng phân tách)** | **85.01%** | 84.31% | 🔻 -0.70% | **LR-C2 mạnh mẽ hơn**: Đạt khả năng phân biệt bệnh nhân đột quỵ và người bình thường tốt nhất toàn dự án. |
| **Tính giải thích Y khoa** | **Tuyệt vời** *(Odds Ratio)* | Trung bình *(Feature Importance)* | - | **LR-C2 vượt trội**: Dễ dàng diễn giải các hệ số tác động lâm sàng cho bác sĩ điều trị và bệnh nhân hiểu rõ lý do dự đoán. |

> [!NOTE]
> - Các mô hình sử dụng dữ liệu **Full 11** đạt độ cân bằng F1-Score cao hơn so với **MI (5 Features)** khoảng 2-3%. Tuy nhiên, việc sử dụng 5 đặc trưng MI vẫn giữ được Recall vượt mốc 80% ở hầu hết các thuật toán, là giải pháp tối ưu chi phí xét nghiệm lâm sàng cực kỳ lớn.
> - Hầu hết các cấu hình đạt Recall > 80% đều đến từ chiến lược điều chỉnh ngưỡng nhạy cảm **Extreme Recall** hoặc **Max Recall** (tối ưu hóa ngưỡng quyết định dựa trên PR-Curve của tập Validation).

---

## 🛠️ CHI TIẾT TOP 3 CẤU HÌNH TỐI ƯU CỦA TỪNG THUẬT TOÁN

### 1. LOGISTIC REGRESSION (LR)
Mô hình Hồi quy tuyến tính được tối ưu hóa bằng cách kết hợp `class_weight='balanced'` và kỹ thuật SMOTE trên tập Train, sau đó tinh chỉnh ngưỡng xác suất quyết định trên tập Validation.

#### A. Khi sử dụng Full 11 Features
* **Top 1: LR-C2 (C=0.1 - Tối ưu nhất)**: 
  * *Siêu tham số*: Điều chuẩn vừa phải ($C=0.1$), `penalty='l2'`.
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.4245**).
  * *Hiệu năng*: **Recall = 83.89%**, **Precision = 52.36%**, **F1-Score = 64.47%**, **AUC = 85.01%**.
* **Top 2: LR-C5 (C=100.0 - Ít điều chuẩn)**: 
  * *Siêu tham số*: Phạt điều chuẩn rất thấp ($C=100.0$), `penalty='l2'`.
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.4231**).
  * *Hiệu năng*: **Recall = 83.22%**, **Precision = 52.77%**, **F1-Score = 64.58%**, **AUC = 85.08%**.
* **Top 3: LR-C3 (C=1.0 - Tiêu chuẩn)**: 
  * *Siêu tham số*: Mặc định ($C=1.0$), `penalty='l2'`.
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.4254**).
  * *Hiệu năng*: **Recall = 83.05%**, **Precision = 52.88%**, **F1-Score = 64.62%**, **AUC = 85.06%**.

#### B. Khi sử dụng MI (5 Features)
* **Top 1: LR-C2 (C=0.1 - Tối ưu nhất)**: 
  * *Siêu tham số*: Điều chuẩn vừa phải ($C=0.1$).
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.4444**).
  * *Hiệu năng*: **Recall = 82.38%**, **Precision = 49.10%**, **F1-Score = 61.53%**, **AUC = 81.50%**.
* **Top 2: LR-C3 (C=1.0 - Tiêu chuẩn)**: 
  * *Siêu tham số*: Mặc định ($C=1.0$).
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.4584**).
  * *Hiệu năng*: **Recall = 80.87%**, **Precision = 50.10%**, **F1-Score = 61.87%**, **AUC = 81.53%**.
* **Top 3: LR-C4 (C=10.0 - Phạt thấp)**: 
  * *Siêu tham số*: Ít điều chuẩn ($C=10.0$).
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.4589**).
  * *Hiệu năng*: **Recall = 80.70%**, **Precision = 50.16%**, **F1-Score = 61.87%**, **AUC = 81.54%**.

> [!TIP]
> **Logistic Regression** thể hiện sự ổn định kinh ngạc khi giảm thuộc tính từ 11 xuống 5. F1-score chỉ giảm nhẹ khoảng ~3% trong khi Recall vẫn đảm bảo chắc chắn trên 80.70%. Đây là mô hình tuyến tính đơn giản, dễ giải thích bằng các hệ số Odds Ratio nên cực kỳ thích hợp để triển khai thực tế.

---

### 2. SUPPORT VECTOR MACHINE (SVM)
SVM sử dụng RBF Kernel cho phép phân tách phi tuyến mạnh mẽ trong không gian đặc trưng nâng cao. Tận dụng `class_weight='balanced'` để tự động phạt nặng các sai số thuộc lớp bệnh nhân.

#### A. Khi sử dụng Full 11 Features
* **Top 1: SVM-C5 (C=100.0, g=0.01 - Lực phân tách cao)**: 
  * *Siêu tham số*: Phạt lỗi phân loại rất nặng ($C=100.0$), gamma nhỏ ($0.01$).
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.1788**).
  * *Hiệu năng*: **Recall = 85.07%**, **Precision = 50.20%**, **F1-Score = 63.14%**, **AUC = 84.64%**.
* **Top 2: SVM-C2 (C=1.0, g='scale' - Cân bằng nhất)**: 
  * *Siêu tham số*: C mặc định ($1.0$), tự động gamma.
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.1727**).
  * *Hiệu năng*: **Recall = 84.23%**, **Precision = 52.02%**, **F1-Score = 64.32%**, **AUC = 84.01%**.
* **Top 3: SVM-C4 (C=10.0, g=0.1 - Biên phi tuyến phức tạp)**: 
  * *Siêu tham số*: C trung bình-cao ($10.0$), gamma lớn ($0.1$).
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.1734**).
  * *Hiệu năng*: **Recall = 83.72%**, **Precision = 52.14%**, **F1-Score = 64.26%**, **AUC = 83.64%**.

#### B. Khi sử dụng MI (5 Features)
* **Top 1: SVM-C3 (C=10.0, g=0.01 - Duy nhất đạt Recall > 80%)**: 
  * *Siêu tham số*: $C=10.0$, gamma rất nhỏ ($0.01$).
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.3516**).
  * *Hiệu năng*: **Recall = 80.87%**, **Precision = 50.16%**, **F1-Score = 61.91%**, **AUC = 81.05%**.
* **Top 2: SVM-C5 (C=100.0, g=0.01 - Tiệm cận sát 80%)**: 
  * *Siêu tham số*: $C=100.0$, gamma nhỏ ($0.01$).
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.3020**).
  * *Hiệu năng*: **Recall = 79.36%**, **Precision = 50.97%**, **F1-Score = 62.07%**, **AUC = 80.23%**.
* **Top 3: SVM-C4 (C=10.0, g=0.1 - Tiệm cận 79%)**: 
  * *Siêu tham số*: $C=10.0$, gamma trung bình ($0.1$).
  * *Chiến lược ngưỡng*: Max Recall (Ngưỡng = **0.2428**).
  * *Hiệu năng*: **Recall = 78.86%**, **Precision = 51.09%**, **F1-Score = 62.01%**, **AUC = 79.13%**.

> [!WARNING]
> Trên dữ liệu rút gọn MI, SVM gặp khó khăn hơn do không gian đặc trưng bị thu hẹp đáng kể, khiến ranh giới quyết định phi tuyến kém mịn hơn. Chỉ có duy nhất cấu hình `SVM-C3` vượt ngưỡng 80% Recall. Tuy nhiên, 2 cấu hình còn lại vẫn cho F1-Score rất khả quan (> 62%) với Recall tiệm cận sát sạt mục tiêu (~79%).

---

### 3. RANDOM FOREST (RF)
Mô hình Rừng ngẫu nhiên chống quá khớp tốt nhờ cơ chế học tập dựa trên Ensemble (Bagging). Sử dụng `class_weight='balanced'` giúp cân bằng các cây quyết định độc lập.

#### A. Khi sử dụng Full 11 Features
* **Top 1: RF-C2 (max_depth=6 - Tối ưu hóa độ sâu)**: 
  * *Siêu tham số*: Cây nông vừa phải (`max_depth=6`), `n_estimators=100`.
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.4401**).
  * *Hiệu năng*: **Recall = 82.89%**, **Precision = 52.61%**, **F1-Score = 64.36%**, **AUC = 84.70%**.
* **Top 2: RF-C3 (max_depth=8 - Tiêu chuẩn)**: 
  * *Siêu tham số*: Cây tiêu chuẩn (`max_depth=8`), `n_estimators=200`.
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.4023**).
  * *Hiệu năng*: **Recall = 82.38%**, **Precision = 51.96%**, **F1-Score = 63.72%**, **AUC = 84.52%**.
* **Top 3: RF-C1 (max_depth=4 - Cân bằng F1 tốt nhất)**: 
  * *Siêu tham số*: Cây rất nông (`max_depth=4`), `n_estimators=50`.
  * *Chiến lược ngưỡng*: Max Recall (Ngưỡng = **0.4688**).
  * *Hiệu năng*: **Recall = 81.54%**, **Precision = 53.88%**, **F1-Score = 64.89%**, **AUC = 84.31%**.

#### B. Khi sử dụng MI (5 Features)
* **Top 1: RF-C4 (max_depth=8, split=10 - Tối ưu nhất)**: 
  * *Siêu tham số*: `max_depth=8`, `n_estimators=300`, `min_samples_split=10`.
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.4092**).
  * *Hiệu năng*: **Recall = 81.04%**, **Precision = 49.24%**, **F1-Score = 61.26%**, **AUC = 80.39%**.
* **Top 2: RF-C5 (max_depth=12 - Cây sâu, phức tạp)**: 
  * *Siêu tham số*: Cây sâu và nặng (`max_depth=12`), `n_estimators=500`.
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.3421**).
  * *Hiệu năng*: **Recall = 80.54%**, **Precision = 46.97%**, **F1-Score = 59.33%**, **AUC = 79.04%**.
* **Top 3: RF-C2 (max_depth=6 - Cây trung bình)**: 
  * *Siêu tham số*: `max_depth=6`, `n_estimators=100`.
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.4268**).
  * *Hiệu năng*: **Recall = 80.20%**, **Precision = 49.59%**, **F1-Score = 61.28%**, **AUC = 80.92%**.

> [!TIP]
> **Random Forest** chứng tỏ vị thế xuất sắc khi tất cả các cấu hình được chọn đều vượt mốc Recall > 80% trên cả 2 tập dữ liệu. Đặc biệt, cấu hình `RF-C1` trên tập Full 11 đạt **F1-Score cao nhất toàn dự án (64.89%)** cùng Recall ổn định 81.54% nhờ việc khống chế độ sâu cây ở mức 4, giúp mô hình giữ được tính tổng quát hóa cao.

---

### 4. XGBOOST (XGB)
XGBoost là thuật toán Gradient Boosting cực kỳ mạnh mẽ. Do XGBoost mặc định không có class_weight trực tiếp giống Scikit-learn (phải dùng scale_pos_weight nhưng dễ gây mất ổn định ngưỡng), hệ thống sử dụng cơ chế SMOTE kết hợp tối ưu hóa ngưỡng xác suất đầu ra để đẩy mạnh Recall.

#### A. Khi sử dụng Full 11 Features
* **Top 1: XGB-C4 (depth=6, lr=0.1 - Học nhanh, sâu)**: 
  * *Siêu tham số*: Cây tiêu chuẩn (`max_depth=6`), `learning_rate=0.1`, `n_estimators=300`.
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.1679**).
  * *Hiệu năng*: **Recall = 84.23%**, **Precision = 47.36%**, **F1-Score = 60.63%**, **AUC = 81.09%**.
* **Top 2: XGB-C5 (depth=10, lr=0.01 - Rất sâu, học chậm)**: 
  * *Siêu tham số*: Cây rất sâu (`max_depth=10`), `learning_rate=0.01`, `n_estimators=500`.
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.1774**).
  * *Hiệu năng*: **Recall = 83.72%**, **Precision = 47.39%**, **F1-Score = 60.52%**, **AUC = 82.27%**.
* **Top 3: XGB-C1 (depth=3, lr=0.1 - Nông, hiệu năng đỉnh cao)**: 
  * *Siêu tham số*: Cây rất nông (`max_depth=3`), `learning_rate=0.1`, `n_estimators=100`.
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.2581**).
  * *Hiệu năng*: **Recall = 82.55%**, **Precision = 52.62%**, **F1-Score = 64.27%**, **AUC = 84.72%**.

#### B. Khi sử dụng MI (5 Features)
* **Top 1: XGB-C5 (depth=10, lr=0.01 - Tối đa Recall)**: 
  * *Siêu tham số*: `max_depth=10`, `learning_rate=0.01`, `n_estimators=500`.
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.2108**).
  * *Hiệu năng*: **Recall = 83.22%**, **Precision = 45.38%**, **F1-Score = 58.73%**, **AUC = 77.72%**.
* **Top 2: XGB-C4 (depth=6, lr=0.1 - Học nhanh)**: 
  * *Siêu tham số*: `max_depth=6`, `learning_rate=0.1`, `n_estimators=300`.
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.2035**).
  * *Hiệu năng*: **Recall = 82.55%**, **Precision = 46.02%**, **F1-Score = 59.10%**, **AUC = 78.07%**.
* **Top 3: XGB-C2 (depth=5, lr=0.05 - Cân bằng tốt)**: 
  * *Siêu tham số*: `max_depth=5`, `learning_rate=0.05`, `n_estimators=200`.
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.2633**).
  * *Hiệu năng*: **Recall = 81.21%**, **Precision = 47.92%**, **F1-Score = 60.27%**, **AUC = 79.87%**.

> [!TIP]
> Với **XGBoost**, cấu hình cây nông `XGB-C1` trên tập Full 11 vượt trội hoàn toàn về mặt cân bằng so với các cấu hình cây sâu. Nó giữ được F1-Score ở mức **64.27%** trong khi vẫn mang lại Recall rất cao **82.55%** và AUC xuất sắc **84.72%**. Việc hạn chế độ sâu cây giúp XGBoost kiểm soát tốt overfitting đối với lớp thiểu số đột quỵ.

---

### 5. DECISION TREE (DT)
Mô hình Cây quyết định đơn lẻ dễ bị quá khớp nhưng có tốc độ huấn luyện và suy luận nhanh nhất, đồng thời có tính trực quan cao nhất bằng sơ đồ cây.

#### A. Khi sử dụng Full 11 Features
* **Top 1: DT-C1 (depth=3 - Siêu nông, Recall cực đại)**: 
  * *Siêu tham số*: Cây siêu nông (`max_depth=3`), `min_samples_split=2`.
  * *Chiến lược ngưỡng*: Max Recall (Ngưỡng = **0.2911**).
  * *Hiệu năng*: **Recall = 92.45%**, **Precision = 44.65%**, **F1-Score = 60.22%**, **AUC = 82.03%**.
* **Top 2: DT-C2 (depth=5 - Tầm trung)**: 
  * *Siêu tham số*: Cây trung bình (`max_depth=5`), `min_samples_split=5`.
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.2054**).
  * *Hiệu năng*: **Recall = 91.95%**, **Precision = 43.25%**, **F1-Score = 58.83%**, **AUC = 82.87%**.
* **Top 3: DT-C3 (depth=6, leaf=20 - Đỉnh cao cân bằng)**: 
  * *Siêu tham số*: Cây hạn chế số lá tối thiểu (`max_depth=6`, `min_samples_leaf=20`).
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.2500**).
  * *Hiệu năng*: **Recall = 85.07%**, **Precision = 47.92%**, **F1-Score = 61.31%**, **AUC = 82.87%**.

#### B. Khi sử dụng MI (5 Features)
* **Top 1: DT-C1 (depth=3 - Siêu nông)**: 
  * *Siêu tham số*: `max_depth=3`, `min_samples_split=2`.
  * *Chiến lược ngưỡng*: Max Recall (Ngưỡng = **0.2835**).
  * *Hiệu năng*: **Recall = 90.10%**, **Precision = 44.79%**, **F1-Score = 59.83%**, **AUC = 80.08%**.
* **Top 2: DT-C5 (depth=12 - Cây sâu, balanced)**: 
  * *Siêu tham số*: Cây sâu và phân mảnh (`max_depth=12`, `min_samples_split=20`, `class_weight='balanced'`).
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.1818**).
  * *Hiệu năng*: **Recall = 86.24%**, **Precision = 41.49%**, **F1-Score = 56.02%**, **AUC = 74.03%**.
* **Top 3: DT-C4 (depth=8 - Cây sâu vừa phải)**: 
  * *Siêu tham số*: `max_depth=8`, `min_samples_split=10`, `class_weight='balanced'`.
  * *Chiến lược ngưỡng*: Extreme Recall (Ngưỡng = **0.3872**).
  * *Hiệu năng*: **Recall = 85.07%**, **Precision = 44.63%**, **F1-Score = 58.55%**, **AUC = 76.74%**.

> [!CAUTION]
> **Hiện tượng sụt giảm ranh giới của Decision Tree**:
> Mặc dù các cấu hình nông như `DT-C1` và `DT-C2` đạt mức Recall cực khủng (**92.45%** và **90.10%**), Precision của chúng bị kéo tụt xuống mức khá thấp (43% - 44%). Điều này là do cây quyết định phân tách không gian một cách thô (bằng các trục vuông góc), dẫn đến việc gắn nhãn sai hàng loạt mẫu khỏe mạnh thành có nguy cơ đột quỵ ở các nhánh lá rìa. Trong ứng dụng thực tế, nên ưu tiên cấu hình pruned `DT-C3` (F1 = 61.31%, Recall = 85.07%) để đảm bảo độ tin cậy y tế tốt hơn.

---

## 📈 KẾT LUẬN & KHUYẾN NGHỊ TRIỂN KHAI LÂM SÀNG

Từ kết quả thực nghiệm chi tiết trên, chúng ta rút ra các khuyến nghị chiến lược quan trọng sau:

### 1. Mô hình lựa chọn tối ưu nhất cho triển khai thực tế (Golden Standard)
Nếu bệnh viện có đầy đủ nguồn lực thu thập 11 thông số đầu vào của bệnh nhân:
* **Khuyến nghị sử dụng**: **Logistic Regression (LR-C2)** hoặc **Random Forest (RF-C1)**.
* **Lý do**:
  * **RF-C1** đạt F1-Score cao nhất toàn bộ dự án (**64.89%**), kiểm soát báo động giả tốt nhất (Precision đạt **53.88%** - mức tốt nhất khi tối ưu hóa Recall > 80%) và giữ mức Recall cực kỳ an toàn (**81.54%**).
  * **LR-C2** có tính giải thích y khoa (Odds Ratio) vượt trội hơn RF, đồng thời cho Recall cao hơn (**83.89%**) và F1-Score vô cùng vững chãi (**64.47%**).

### 2. Sự đánh đổi hiệu quả khi áp dụng Giảm chiều đặc trưng (Mutual Information)
Khi rút gọn từ 11 đặc trưng xuống còn 5 đặc trưng lâm sàng cốt lõi (`Age`, `Hypertension`, `Heart_Disease`, `Avg_Glucose`, `SES`):
* **Lợi ích**:
  * **Tiết kiệm chi phí**: Bác sĩ chỉ cần thu thập thông tin về độ tuổi, tiền sử cao huyết áp/bệnh tim, đo đường huyết nhanh và xác định tình trạng SES (không cần đo đạc hay phỏng vấn chi tiết các thói quen hút thuốc phức tạp khác).
  * **Tiết kiệm thời gian**: Phù hợp cho các phòng cấp cứu hoặc sàng lọc nhanh tại cộng đồng.
* **Chi phí đánh đổi**:
  * **Độ nhạy (Recall)** chỉ bị sụt giảm cực nhẹ (trung bình giảm khoảng 1.5% - 2%, vẫn giữ vững mốc > 80% an toàn).
  * **F1-Score** giảm từ khoảng ~64% xuống ~61% (Precision giảm nhẹ, tăng thêm khoảng 3-4% số ca báo động giả).
* **Kết luận chiến lược**: Việc giảm chiều đặc trưng qua **Mutual Information** đạt hiệu quả thực tế cực kỳ cao, rất khuyến khích ứng dụng vào các hệ thống sàng lọc nhanh ban đầu (First-line Screening Tool).
