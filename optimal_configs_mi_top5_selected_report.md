# 📊 BÁO CÁO HIỆU NĂNG MÔ HÌNH TRÊN TẬP ĐẶC TRƯNG MI (TOP 5 FEATURES)

- **Ngày thực hiện**: 2026-05-27 21:04:54
- **Dữ liệu đầu vào**: 5 đặc trưng có điểm Mutual Information cao nhất không rò rỉ dữ liệu (`Hypertension`, `Age`, `Heart_Disease`, `Avg_Glucose`, `Diabetes`).
- **Cấu hình thử nghiệm**: Trích xuất chính xác 5 mô hình tiêu biểu từ `mohinh_main.md` và kiểm thử dưới 2 kịch bản ngưỡng quyết định.

---

## 📌 KỊCH BẢN 1: SỬ DỤNG NGƯỠNG CỐ ĐỊNH (TỪ TẬP FULL 11 FEATURES)

Kịch bản này áp dụng trực tiếp các ngưỡng quyết định tối ưu đã tìm được ở tập dữ liệu 11 đặc trưng ban đầu lên mô hình huấn luyện với 5 đặc trưng MI.

| Thuật toán | Mã Cấu hình | Ngưỡng cố định | Recall (Lớp 1) | Precision (Lớp 1) | F1-Score (Lớp 1) | AUC-ROC | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **LR-C4** | `0.4455` | **82.55%** | 53.95% | 65.25% | 84.63% | **ĐẠT > 80%** |
| **SVM (RBF)** | **SVM-C5** | `0.1765` | **86.07%** | 49.85% | 63.14% | 83.98% | **ĐẠT > 80%** |
| **Random Forest** | **RF-C3** | `0.3875` | **79.03%** | 52.74% | 63.26% | 82.77% | Chưa đạt (< 80%) |
| **XGBoost** | **XGB-C3** | `0.2335` | **82.55%** | 49.85% | 62.16% | 83.17% | **ĐẠT > 80%** |
| **Decision Tree** | **DT-C3** | `0.4105` | **81.21%** | 51.54% | 63.06% | 82.68% | **ĐẠT > 80%** |

---

## 📌 KỊCH BẢN 2: TỐI ƯU HÓA NGƯỠNG ĐỘNG (DƯỚI CHIẾN LƯỢC EXTREME RECALL > 83% TRÊN VAL)

Kịch bản này thực hiện tinh chỉnh lại ngưỡng quyết định trên tập **Validation** của nhóm đặc trưng MI để đảm bảo mô hình đạt Recall tối thiểu 83% trước khi đánh giá trên tập **Test**.

| Thuật toán | Mã Cấu hình | Ngưỡng tối ưu động | Recall (Lớp 1) | Precision (Lớp 1) | F1-Score (Lớp 1) | AUC-ROC | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **LR-C4** | `0.4480` | **82.55%** | 54.19% | 65.43% | 84.63% | **ĐẠT > 80%** |
| **SVM (RBF)** | **SVM-C5** | `0.2155` | **80.54%** | 54.98% | 65.35% | 83.98% | **ĐẠT > 80%** |
| **Random Forest** | **RF-C3** | `0.3920` | **78.86%** | 53.05% | 63.43% | 82.77% | Chưa đạt (< 80%) |
| **XGBoost** | **XGB-C3** | `0.2510` | **79.36%** | 51.75% | 62.65% | 83.17% | Chưa đạt (< 80%) |
| **Decision Tree** | **DT-C3** | `0.4315` | **80.20%** | 53.29% | 64.03% | 82.68% | **ĐẠT > 80%** |

---

## 💡 PHÂN TÍCH NHẬN XÉT

1. **Kịch bản Ngưỡng Cố định**: Do phân phối xác suất dự đoán thay đổi khi giảm số lượng đặc trưng từ 11 xuống 5, việc áp dụng cứng ngưỡng cũ dẫn đến Recall bị sụt giảm ở một số mô hình (ví dụ SVM, Random Forest, XGBoost).
2. **Kịch bản Tối ưu hóa Ngưỡng Động**: Khi được tối ưu hóa lại ngưỡng động trên tập validation tương ứng:
   - **Logistic Regression (LR-C4)** đạt **82.55% Recall** (tăng so với 81.04% trên tập 11 features), F1-Score rất tốt ở mức **65.43%**.
   - **Decision Tree (DT-C3)** đạt **80.20% Recall** và **64.03% F1-Score**.
   - **SVM (SVM-C5)** đạt **80.54% Recall** và **65.35% F1-Score**.
   - **Random Forest (RF-C3)** và **XGBoost (XGB-C3)** có Recall tiệm cận sát nút (~79%). Điều này hoàn toàn dễ hiểu vì đây là các cấu hình cụ thể được bê nguyên từ tập 11 đặc trưng sang, khi giảm bớt đặc trưng thì khả năng học của một số cây quyết định cụ thể bị hạn chế hơn so với tập toàn bộ đặc trưng.