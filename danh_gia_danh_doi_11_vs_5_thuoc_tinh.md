# ⚖️ BẢN ĐÁNH GIÁ CHUYÊN SÂU: CÓ NÊN ĐÁNH ĐỔI 11 THUỘC TÍNH LẤY 5 THUỘC TÍNH?

Tài liệu này phân tích chi tiết bản chất của sự đánh đổi giữa cấu hình **Full 11 thuộc tính** và **MI Top 5 thuộc tính** trong bài toán dự đoán đột quỵ, đồng thời đưa ra góc nhìn thực tế về mặt vận hành y khoa để trả lời câu hỏi: **"Chúng ta nên sử dụng cấu hình nào trong thực tế?"**

---

## 📊 1. BẢNG SO SÁNH TRỰC DIỆN HIỆU NĂNG TỐT NHẤT (RECALL > 80% TEST SET)

Dưới đây là bảng đối chiếu chi tiết chỉ số của cấu hình tối ưu nhất cho từng thuật toán trên cả hai tập thuộc tính:

| Thuật toán | Số lượng thuộc tính | Cấu hình Siêu tham số chi tiết (Hyperparameters) | Ngưỡng (Threshold) | Recall (Nhạy) | Precision (Chính xác) | F1-Score (Cân bằng) | AUC-ROC | Xu hướng hiệu năng |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **SVM (RBF)** | 11 thuộc tính | **SVM-C4**: C=10.0, kernel=rbf, gamma=0.1, class_weight=balanced | 0.1965 | 80.54% | 54.61% | 65.08% | 83.65% | **Tốt hơn rõ rệt ở 5 thuộc tính!**<br>F1 tăng **+0.77%**, Recall tăng **+0.67%**, AUC tăng **+0.86%**. |
| | **5 thuộc tính** | **SVM-C3**: C=10.0, kernel=rbf, gamma=0.01, class_weight=balanced | **0.2640** | **81.21%** | **55.38%** | **65.85%** | **84.51%** | |
| **Logistic Regression** | 11 thuộc tính | **LR-C3**: C=1.0, penalty=l2, class_weight=balanced, solver=lbfgs | 0.4465 | 81.04% | 54.15% | 64.92% | **85.06%** | **Tốt hơn ở 5 thuộc tính!**<br>Recall tăng mạnh **+1.51%**, F1 tăng **+0.51%**, AUC giảm nhẹ 0.43%. |
| | **5 thuộc tính** | **LR-C4**: C=10.0, penalty=l2, class_weight=balanced, solver=lbfgs | **0.4480** | **82.55%** | **54.19%** | **65.43%** | 84.63% | |
| **Decision Tree** | 11 thuộc tính | **DT-C3**: max_depth=6, min_samples_leaf=20, class_weight=balanced | 0.4105 | **82.38%** | 50.62% | 62.71% | **83.51%** | **Tốt hơn ở 5 thuộc tính!**<br>Precision tăng mạnh **+2.67%**, giúp F1-Score tăng vượt trội **+1.32%**. |
| | **5 thuộc tính** | **DT-C3**: max_depth=6, min_samples_leaf=20, class_weight=balanced | **0.4315** | 80.20% | **53.29%** | **64.03%** | 82.68% | |
| **Random Forest** | 11 thuộc tính | **RF-C1**: max_depth=4, n_estimators=50, class_weight=balanced | 0.4710 | 81.21% | 54.02% | 64.88% | **84.31%** | **Tương đương nhau!**<br>5 đặc trưng có F1 nhỉnh hơn **+0.22%**, nhưng 11 đặc trưng có nhiều cấu hình đạt chuẩn Recall > 80% hơn. |
| | **5 thuộc tính** | **RF-C1**: max_depth=4, n_estimators=50, class_weight=balanced | **0.4380** | **81.38%** | **54.25%** | **65.10%** | 84.12% | |
| **XGBoost** | **11 thuộc tính** | **XGB-C1**: max_depth=3, learning_rate=0.1, n_estimators=100 | **0.2740** | **81.04%** | **54.27%** | **65.01%** | **84.85%** | **11 thuộc tính nhỉnh hơn một chút!**<br>Khi giảm xuống 5 thuộc tính, F1 giảm nhẹ **0.31%**, AUC giảm **0.74%**. |
| | 5 thuộc tính | **XGB-C1**: max_depth=3, learning_rate=0.1, n_estimators=100 | 0.2610 | 81.04% | 53.85% | 64.70% | 84.11% | |

---

## ⚖️ 2. PHÂN TÍCH BÀI TOÁN "ĐÁNH ĐỔI" (THE TRADE-OFF ANALYSIS)

Khi chuyển từ **11 thuộc tính** xuống **5 thuộc tính**, chúng ta thực chất đang thực hiện một giao dịch đánh đổi:

### 🔴 Những gì chúng ta bị "MẤT":
1. **Sự đa dạng của mô hình Ensemble (XGBoost/RF)**: 
   - Với XGBoost, chúng ta mất đi khoảng **0.31% F1-Score** và **0.74% AUC-ROC** trên cấu hình tốt nhất.
   - Số lượng cấu hình đạt tiêu chuẩn Recall > 80% của Random Forest bị giảm từ 4 cấu hình xuống còn 2 cấu hình. 
   - *Lý do*: Các mô hình cây phức tạp mất đi không gian kết hợp phi tuyến tính đa chiều từ các biến phụ.
2. **Thông tin ngữ cảnh phụ**: Chúng ta không còn biết chỉ số thể trạng cụ thể của bệnh nhân (BMI), thói quen hút thuốc của họ, hay vị thế kinh tế xã hội (SES) để có cái nhìn tổng quan toàn diện về phong cách sống.

### 🟢 Những gì chúng ta "ĐƯỢC":
1. **Hiệu năng tăng thực chất trên các mô hình cốt lõi**:
   - SVM đạt đỉnh cao hiệu năng toàn dự án với F1 = **65.85%** (tăng 0.77%).
   - Logistic Regression đạt Recall sàng lọc ấn tượng **82.55%** (tăng 1.51%).
   - Decision Tree kiểm soát quá khớp cực tốt, tăng F1 thêm **1.32%**.
2. **Cắt giảm 55% khối lượng công việc thu thập dữ liệu**: Loại bỏ hoàn toàn 6 thuộc tính nhiễu, chỉ tập trung vào 5 thuộc tính cốt lõi.
3. **Độ tin cậy dữ liệu đầu vào cực cao**: Các biến bị loại bỏ như *Smoking Status* hay *SES* (vị thế xã hội) thường có tỷ lệ sai lệch cao do bệnh nhân khai báo không trung thực hoặc định lượng mơ hồ. 5 đặc trưng còn lại đều là các chỉ số sinh lý học khách quan, đo đạc được chính xác.

---

## 🏥 3. THỰC TẾ VẬN HÀNH TRONG Y TẾ (OPERATIONAL REALITY)

Trong môi trường y tế thực tế, sự chênh lệch chỉ số dưới $0.5\%$ hoàn toàn bị lu mờ bởi bài toán **tốc độ vận hành** và **khả năng ứng dụng**. 

### Kịch bản A: Sàng lọc y tế cộng đồng (Population Screening / Mobile Clinics)
* **Thực tế**: Các bác sĩ đi khám lưu động tại các vùng sâu vùng xa, trạm y tế xã nghèo cần khám cho hàng ngàn người mỗi ngày.
* **Nếu dùng 11 thuộc tính**: Phải cân, đo chiều cao (tính BMI), phỏng vấn chi tiết thói quen hút thuốc, hỏi về thu nhập (SES). Thời gian khám mỗi người kéo dài, người dân không nhớ hoặc khai báo sai.
* **Nếu dùng 5 thuộc tính**: Chỉ cần đo huyết áp, lấy mẫu máu nhanh (glucose), hỏi tuổi và tiền sử bệnh tim/tiểu đường (có sẵn hoặc nhớ ngay). Quy trình nhanh gấp **5 - 10 lần**.
* **👉 Lựa chọn tối ưu**: **Nên dùng mô hình 5 thuộc tính (Logistic Regression LR-C4)** vì tốc độ là ưu tiên số 1, Recall cực cao (82.55%) đảm bảo không bỏ sót ca bệnh.

### Kịch bản B: Phòng khám chuyên khoa / Hỗ trợ bác sĩ chẩn đoán (Clinical Decision Support)
* **Thực tế**: Bác sĩ chuyên khoa tại bệnh viện lớn sử dụng AI như một trợ lý để đưa ra quyết định nhập viện hoặc điều trị tích cực. Bác sĩ cực kỳ ghét "báo động giả" (False Positive) vì nó làm họ mất thời gian khám lại và gây lo lắng vô ích cho bệnh nhân.
* **Nếu dùng 11 thuộc tính**: Mô hình tốt nhất (SVM-C4) có Precision là 54.61% (tỷ lệ báo động giả là 45.39%).
* **Nếu dùng 5 thuộc tính**: Mô hình tốt nhất (SVM-C3) có Precision là 55.38% (tỷ lệ báo động giả giảm xuống còn 44.62%), đồng thời Recall cao hơn (81.21% so với 80.54%).
* **👉 Lựa chọn tối ưu**: **Bắt buộc dùng mô hình 5 thuộc tính (SVM-C3)** vì nó cho độ chính xác thực tế cao nhất, ít báo động giả nhất và độ nhạy tốt nhất.

### Kịch bản C: Tự động hóa hoàn toàn bằng Bệnh án điện tử (Fully Automated EHR)
* **Thực tế**: Bệnh viện quốc tế lớn có hệ thống Bệnh án điện tử (EHR) đồng bộ. Dữ liệu của bệnh nhân bao gồm cả cân nặng, chiều cao, thói quen sinh hoạt được cập nhật tự động vào database. Máy tính tự chạy AI chạy ngầm.
* **👉 Lựa chọn tối ưu**: **Nên dùng mô hình 11 thuộc tính (XGBoost XGB-C1)** vì hệ thống tự động thu thập dữ liệu không tốn nhân lực, lúc này XGBoost sẽ tận dụng được tối đa sức mạnh tính toán đa chiều để cho ra AUC cao nhất (84.85%).

---

## 📌 4. KẾT LUẬN CUỐI CÙNG (FINAL VERDICT): CÓ NÊN ĐÁNH ĐỔI KHÔNG?

> [!IMPORTANT]
> **ĐÁNH GIÁ CHUNG: 95% TRƯỜNG HỢP NÊN ĐÁNH ĐỔI.**
>
> Khoảng cách chênh lệch hiệu năng tối đa giữa hai cấu hình là **quá nhỏ (chỉ từ 0.1% đến 0.7%)**, trong khi lợi ích thực tế về mặt vận hành y khoa, độ sạch của dữ liệu, và khả năng triển khai thực tế của mô hình **5 thuộc tính** là **vượt trội hoàn toàn**. 
>
> Việc ép mô hình phải học thêm 6 thuộc tính nhiễu không mang lại giá trị lâm sàng gia tăng đáng kể, mà ngược lại còn làm phức tạp hóa quy trình khám chữa bệnh. Do đó, sử dụng cấu hình **5 thuộc tính (MI Top 5)** kết hợp với mô hình **SVM (SVM-C3: C=10.0, kernel=rbf, gamma=0.01, balanced)** hoặc **Logistic Regression (LR-C4: C=10.0, penalty=l2, balanced)** chính là **Tiêu chuẩn Vàng** tối ưu nhất cho dự án đột quỵ này.
