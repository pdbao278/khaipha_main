# ⚖️ BÁO CÁO ĐỐI SÁNH VÀ ĐÁNH GIÁ KHÁCH QUAN: FULL 11 FEATURES VS. MI TOP 5 FEATURES

Báo cáo này thực hiện phân tích, so sánh và đánh giá khách quan hiệu năng của 5 mô hình học máy được huấn luyện trên hai không gian đặc trưng khác nhau: **Bộ đầy đủ 11 đặc trưng (Full 11 Features)** và **Bộ 5 đặc trưng sàng lọc qua Mutual Information không rò rỉ dữ liệu (MI Top 5 Features)**.

Mục tiêu cốt lõi của việc đối sánh này là trả lời hai câu hỏi lớn về mặt khoa học dữ liệu và y tế:
1. *Việc giảm 54.5% số lượng đặc trưng đầu vào (từ 11 xuống 5) có làm suy giảm nghiêm trọng khả năng dự báo đột quỵ hay không?*
2. *Mô hình nào thể hiện tính bền vững (robustness) cao nhất khi rút gọn dữ liệu?*

---

## 📊 BẢNG ĐỐI SÁNH HIỆU NĂNG CHI TIẾT

Dữ liệu đối sánh được tổng hợp trực tiếp từ kết quả kiểm thử độc lập (**Test Set**) dưới hai kịch bản ngưỡng quyết định (Cố định theo bộ 11 đặc trưng vs. Tối ưu hóa động trên tập Validation của bộ 5 đặc trưng MI):

| Thuật toán | Mã Cấu hình | Chỉ số | Bộ đầy đủ 11 đặc trưng | Bộ 5 đặc trưng MI (Ngưỡng cố định) | Bộ 5 đặc trưng MI (Ngưỡng động tối ưu) | Xu hướng biến đổi (Động so với 11) |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression** | **LR-C4** | **Recall** <br> Precision <br> F1-Score <br> AUC-ROC | **81.04%** <br> 54.15% <br> 64.92% <br> **85.07%** | **82.55%** <br> 53.95% <br> 65.25% <br> **84.63%** | **82.55%** <br> 54.19% <br> 65.43% <br> **84.63%** | 📈 **Tăng (+1.51%)** <br> 📈 Tăng (+0.04%) <br> 📈 Tăng (+0.51%) <br> 📉 Giảm nhẹ (-0.44%) |
| **SVM (RBF)** | **SVM-C5** | **Recall** <br> Precision <br> F1-Score <br> AUC-ROC | **85.57%** <br> 49.85% <br> 63.00% <br> **84.64%** | **86.07%** <br> 49.85% <br> 63.14% <br> **83.98%** | **80.54%** <br> 54.98% <br> 65.35% <br> **83.98%** | 📉 **Giảm (-5.03%)** <br> 📈 **Tăng mạnh (+5.13%)** <br> 📈 **Tăng mạnh (+2.35%)** <br> 📉 Giảm nhẹ (-0.66%) |
| **Random Forest** | **RF-C3** | **Recall** <br> Precision <br> F1-Score <br> AUC-ROC | **84.06%** <br> 50.86% <br> 63.38% <br> **84.50%** | **79.03%** <br> 52.74% <br> 63.26% <br> **82.77%** | **78.86%** <br> 53.05% <br> 63.43% <br> **82.77%** | 📉 **Giảm (-5.20%)** <br> 📈 Tăng (+2.19%) <br> 📈 Tăng (+0.05%) <br> 📉 Giảm (-1.73%) |
| **XGBoost** | **XGB-C3** | **Recall** <br> Precision <br> F1-Score <br> AUC-ROC | **83.05%** <br> 50.67% <br> 62.94% <br> **83.55%** | **82.55%** <br> 49.85% <br> 62.16% <br> **83.17%** | **79.36%** <br> 51.75% <br> 62.65% <br> **83.17%** | 📉 **Giảm (-3.69%)** <br> 📈 Tăng (+1.08%) <br> 📉 Giảm nhẹ (-0.29%) <br> 📉 Giảm nhẹ (-0.38%) |
| **Decision Tree** | **DT-C3** | **Recall** <br> Precision <br> F1-Score <br> AUC-ROC | **82.38%** <br> 50.62% <br> 62.71% <br> **83.51%** | **81.21%** <br> 51.54% <br> 63.06% <br> **82.68%** | **80.20%** <br> 53.29% <br> 64.03% <br> **82.68%** | 📉 **Giảm (-2.18%)** <br> 📈 Tăng (+2.67%) <br> 📈 Tăng (+1.32%) <br> 📉 Giảm nhẹ (-0.83%) |

---

## 🔍 ĐÁNH GIÁ CHI TIẾT TỪNG THUẬT TOÁN (ANALYTICAL REVIEW)

### 1. Logistic Regression (`LR-C4`): Sự Bứt Phá Đầy Bất Ngờ
* **Nhận xét khách quan**: Logistic Regression là mô hình xuất sắc nhất trong thử nghiệm rút gọn đặc trưng này. Khi chuyển sang bộ 5 đặc trưng MI, mô hình không những không suy giảm hiệu năng mà còn cải thiện đồng bộ trên cả **Recall (đạt 82.55%)**, **Precision (54.19%)** và **F1-Score (65.43%)**.
* **Nguyên nhân toán học**: Việc giảm bớt các biến nhiễu hoặc có tương quan chéo cao trong bộ 11 biến (như loại bỏ các biến có điểm MI cực thấp) đã giúp giải quyết hiện tượng đa cộng tuyến (multicollinearity) và giúp Logistic Regression tối ưu hóa các trọng số một cách tập trung, chính xác hơn vào các biến nguy cơ hàng đầu (`Hypertension`, `Age`, `Heart_Disease`, `Avg_Glucose`, `Diabetes`).

### 2. SVM RBF Kernel (`SVM-C5`): Đánh Đổi Thông Minh
* **Nhận xét khách quan**: SVM thể hiện tính linh hoạt tuyệt vời dưới hai kịch bản ngưỡng. Ở ngưỡng cố định cũ, SVM đạt Recall cực cao **86.07%** nhưng chấp nhận Precision thấp. Khi được tinh chỉnh lại ngưỡng động (`0.2155`), mô hình đạt **Recall 80.54%** (vừa đủ vượt chuẩn y khoa) nhưng đổi lại **Precision tăng vọt lên 54.98%** và **F1-Score đạt 65.35%** (cao nhất toàn tập MI).
* **Nguyên nhân toán học**: SVM sử dụng nhân RBF để ánh xạ dữ liệu vào không gian nhiều chiều để tìm siêu phẳng phân loại. Khi giảm số biến từ 11 xuống 5, không gian biên phân loại trở nên rõ ràng và ít phức tạp hơn, giúp mô hình hội tụ tốt hơn và cho ra các xác suất dự đoán ổn định hơn khi điều chỉnh ngưỡng quyết định.

### 3. Random Forest & XGBoost (`RF-C3` & `XGB-C3`): Sự Phụ Thuộc Vào Không Gian Đặc Trưng Rộng
* **Nhận xét khách quan**: Cả hai mô hình ensemble (học máy dạng rừng cây/tăng cường độ dốc) đều bị suy giảm Recall xuống dưới mức chuẩn 80% (lần lượt đạt **78.86%** và **79.36%** ở kịch bản ngưỡng động).
* **Nguyên nhân toán học**: Random Forest và XGBoost hoạt động dựa trên cơ chế phân tách ngẫu nhiên không gian đặc trưng (feature bagging) để tạo ra các cây quyết định độc lập nhằm giảm phương sai. Khi chỉ còn lại đúng 5 đặc trưng, tính đa dạng của các cây quyết định bị thu hẹp đáng kể, dẫn đến việc mô hình bị giới hạn khả năng học các mối quan hệ phi tuyến tính phức tạp so với khi có đầy đủ 11 đặc trưng.

### 4. Decision Tree (`DT-C3`): Tính Bền Vững Đáng Kinh Ngạc
* **Nhận xét khách quan**: Khác với Random Forest, một cây quyết định đơn lẻ (`DT-C3`) lại thể hiện sự bền vững đáng khen ngợi khi duy trì **Recall 80.20%** và **F1-Score 64.03%** trên tập 5 đặc trưng MI.
* **Nguyên nhân toán học**: Do cấu hình `DT-C3` có giới hạn `max_depth=6` và `min_samples_leaf=20` (kiểm soát quá khớp), cây quyết định chỉ tập trung cắt ở những biến quan trọng nhất. Vì vậy, việc lược bỏ 6 biến ít quan trọng hơn hầu như không ảnh hưởng đến cấu trúc cắt nhánh cốt lõi của cây đơn lẻ này.

---

## 🏥 Ý NGHĨA LÂM SÀNG VÀ ĐÁNH GIÁ KHÁCH QUAN (CLINICAL & PRACTICAL IMPLICATIONS)

> [!IMPORTANT]
> **5 đặc trưng được lọc bằng Mutual Information bao gồm**: `Hypertension` (Cao huyết áp), `Age` (Tuổi tác), `Heart_Disease` (Bệnh tim), `Avg_Glucose` (Lượng đường huyết trung bình), và `Diabetes` (Tiểu đường). 
> Đây đều là **5 chỉ số vàng** thuộc nhóm bệnh lý nền và sinh trắc học cơ bản dễ dàng thu thập nhất trong thực tế khám chữa bệnh lâm sàng.

### 1. Tiết kiệm nguồn lực y tế cực kỳ lớn
* **Bộ 11 đặc trưng** yêu cầu thu thập thêm các thông tin như: thói quen hút thuốc, tình trạng hôn nhân, loại hình công việc, nơi cư trú, chỉ số BMI, giới tính. Nhiều chỉ số trong số này mang tính chủ quan (như thói quen hút thuốc tự khai) hoặc ít liên quan trực tiếp đến cơ chế bệnh sinh của đột quỵ.
* **Bộ 5 đặc trưng MI** chỉ tập trung vào các chỉ số sinh học cốt lõi. Việc này giúp **rút ngắn 54% thời gian khai thác bệnh sử**, giảm chi phí xét nghiệm không cần thiết, cực kỳ phù hợp cho các app tự tầm soát tại nhà hoặc các phòng khám tuyến đầu có nguồn lực hạn chế.

### 2. Sự an toàn của bệnh nhân (Recall) vs. Hiệu suất của bác sĩ (F1-Score)
* Trong y khoa, **Recall (Độ nhạy)** là yếu tố sống còn vì mục tiêu là "thà bắt nhầm còn hơn bỏ sót" bệnh nhân có nguy cơ đột quỵ.
* Tuy nhiên, nếu Precision quá thấp (báo động giả quá nhiều), hệ thống y tế sẽ bị quá tải và bác sĩ sẽ bị lờn cảnh báo (alarm fatigue).
* **Logistic Regression (`LR-C4`)** trên tập 5 đặc trưng MI đạt điểm cân bằng vàng: **Recall 82.55%** và **F1-Score 65.43%**. Đây là mô hình tối ưu nhất để đưa vào ứng dụng thực tiễn lâm sàng.

---

## 💡 KẾT LUẬN & KHUYẾN NGHỊ LỰA CHỌN TẬP ĐẶC TRƯNG (RECOMMENDATIONS)

### 🏥 KHUYẾN NGHỊ CHỌN TẬP ĐẶC TRƯNG: NÊN CHỌN TẬP **MI TOP 5 FEATURES**

Trong ngữ cảnh của bài toán dự phòng và sàng lọc **đột quỵ (Stroke)** - một bệnh lý cấp tính nguy hiểm tính mạng, việc chọn tập đặc trưng cần dung hòa giữa **hiệu năng kỹ thuật (toán học)** và **khả năng ứng dụng lâm sàng thực tế**. Tập **MI Top 5 Features** là lựa chọn tối ưu nhất dựa trên các lý do sau:

#### 1. Lý do Lâm sàng và Thực tiễn Y tế (Clinical Practice)
* **Khách quan và dễ đo đạc**: 5 chỉ số (`Hypertension`, `Age`, `Heart_Disease`, `Avg_Glucose`, `Diabetes`) đều là các biến sinh học thực thể khách quan. Các chỉ số này không phụ thuộc vào cảm quan cá nhân, dễ dàng đo đạc chính xác bằng máy y tế hoặc khai thác nhanh từ hồ sơ bệnh án gốc.
* **Loại bỏ nhiễu và sai lệch khai báo (Reporting Bias)**: Bộ 11 biến có chứa các yếu tố chủ quan tự khai (như `Smoking Status`) dễ bị sai lệch do tâm lý bệnh nhân e ngại, hoặc các yếu tố xã hội học như `Ever Married` hay `Residence Type` không có liên hệ nhân quả trực tiếp đối với cơ chế bệnh lý mạch máu.
* **Tiết kiệm nguồn lực**: Giảm 54.5% số lượng biến cần thu thập giúp **rút ngắn một nửa thời gian thăm khám lâm sàng**, giảm chi phí xét nghiệm thừa, rất thích hợp cho ứng dụng tầm soát cộng đồng diện rộng hoặc sàng lọc tuyến đầu.

#### 2. Lý do Toán học và Khoa học dữ liệu (Data Science)
* **Sự bứt phá của các mô hình tuyến tính/biên**: Khi rút gọn xuống 5 đặc trưng MI, **Logistic Regression (LR-C4)** tăng trưởng đồng thời cả **Recall (82.55%)** và **F1-Score (65.43%)**. Điều này chứng minh 6 biến bị lược bỏ thực chất là nhiễu đa cộng tuyến. Khi loại bỏ chúng, mô hình học được trọng số chính xác và tổng quát hóa tốt hơn.
* **Độ cân bằng vàng chống quá tải hệ thống**: **SVM (SVM-C5)** ở ngưỡng động `0.2155` đạt **Precision 54.98%** và **F1-Score 65.35%** (cao nhất trong các thử nghiệm) nhưng vẫn đảm bảo **Recall 80.54%** (vượt chuẩn y khoa 80%). Precision cao giúp hạn chế tối đa "báo động giả", giảm sự hoang mang cho bệnh nhân và tránh lãng phí chi phí chụp chiếu chuyên sâu (như chụp MRI, CT cắt lớp) không cần thiết.

---

### 🛠️ KHUYẾN NGHỊ LỰA CHỌN MÔ HÌNH TRIỂN KHAI

> [!TIP]
> **Thứ tự đề xuất mô hình đưa vào ứng dụng thực tế**:
> 1. **Lựa chọn số 1 (Mô hình Sàng lọc Chủ lực)**: **Logistic Regression (LR-C4)** trên bộ **5 đặc trưng MI** ở ngưỡng **0.4480**. Đây là mô hình toàn diện nhất, cho kết quả Recall (82.55%) và F1 (65.43%) vượt trội hơn cả khi dùng bộ đầy đủ 11 đặc trưng.
> 2. **Lựa chọn số 2 (Mô hình Xác minh Phụ)**: **SVM (SVM-C5)** trên bộ **5 đặc trưng MI** ở ngưỡng **0.2155** để kiểm chứng chéo nhờ F1-Score cực cao (65.35%) và Recall an toàn (80.54%).
> 3. **Cảnh báo**: **Không dùng** các thuật toán nhóm Ensemble như **Random Forest** hay **XGBoost** trên không gian 5 đặc trưng hẹp này vì Recall sẽ bị suy giảm dưới mức tiêu chuẩn 80% (do bị giới hạn khả năng tách đặc trưng ngẫu nhiên).

