# 📊 Kết Quả Chạy 5 Thuật Toán Phân Loại Đột Quỵ

> Tất cả các mô hình đã được huấn luyện trên dữ liệu phân chia **7/1/2** (Train 70% + SMOTE 0.5, Val 10%, Test 20%) với **2 chiến lược ngưỡng** (Max Recall & Balance F1) và **2 tập đặc trưng** (Full 11 & MI 5).

---

## 🏆 TOP 5 RECALL (Full 11 Features, Max Recall Strategy)

| Model | Config | Threshold | Recall | Precision | F1-Score | AUC-ROC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Decision Tree** | DT-C1 | 0.2911 | **92.45%** | 44.65% | 60.22% | 82.03% |
| **XGBoost** | XGB-C5 | 0.1929 | **81.71%** | 48.17% | 60.61% | 82.26% |
| **Random Forest** | RF-C1 | 0.4712 | **81.21%** | 54.02% | 64.88% | 84.31% |
| **XGBoost** | XGB-C4 | 0.2069 | **79.36%** | 49.84% | 61.23% | 81.36% |
| **Decision Tree** | DT-C3 | 0.2812 | **79.36%** | 52.61% | 63.28% | 82.93% |

---

## 🎯 TOP 5 F1-SCORE (Full 11 Features, Balance F1 Strategy)

| Model | Config | Threshold | Recall | Precision | F1-Score | AUC-ROC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | LR-C3 | 0.5546 | 72.99% | 62.32% | **67.23%** | 85.06% |
| **Logistic Regression** | LR-C5 | 0.5530 | 73.15% | 62.11% | **67.18%** | 85.08% |
| **Logistic Regression** | LR-C2 | 0.5594 | 71.64% | 62.79% | **66.93%** | 85.01% |
| **SVM** | SVM-C4 | 0.2864 | 73.99% | 59.68% | **66.07%** | 83.65% |
| **SVM** | SVM-C5 | 0.4053 | 68.96% | 63.23% | **65.97%** | 84.64% |

---

## 📋 Kết Quả Chi Tiết Từng Thuật Toán

### 1. Logistic Regression

| Config | Feature Set | Strategy | Threshold | Recall | Precision | F1 | AUC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LR-C2 | Full 11 | Max Recall | 0.4894 | 77.68% | 56.12% | 65.17% | 85.01% |
| LR-C2 | Full 11 | Balance F1 | 0.5594 | 71.64% | 62.79% | 66.93% | 85.01% |
| LR-C5 | Full 11 | Max Recall | 0.4796 | 78.02% | 56.16% | 65.31% | 85.08% |
| LR-C5 | Full 11 | Balance F1 | 0.5530 | 73.15% | 62.11% | 67.18% | 85.08% |
| LR-C3 | Full 11 | Max Recall | 0.4933 | 76.51% | 56.93% | 65.28% | 85.06% |
| LR-C3 | Full 11 | Balance F1 | 0.5546 | 72.99% | 62.32% | 67.23% | 85.06% |
| LR-C2 | MI 5 | Max Recall | 0.5119 | 73.15% | 52.59% | 61.19% | 81.48% |
| LR-C2 | MI 5 | Balance F1 | 0.5219 | 72.82% | 53.58% | 61.74% | 81.48% |
| LR-C3 | MI 5 | Max Recall | 0.5001 | 74.66% | 52.29% | 61.51% | 81.53% |
| LR-C3 | MI 5 | Balance F1 | 0.5145 | 73.15% | 53.37% | 61.71% | 81.53% |
| LR-C4 | MI 5 | Max Recall | 0.5002 | 74.66% | 52.35% | 61.55% | 81.53% |
| LR-C4 | MI 5 | Balance F1 | 0.5147 | 73.15% | 53.43% | 61.76% | 81.53% |

### 2. SVM (RBF Kernel)

| Config | Feature Set | Strategy | Threshold | Recall | Precision | F1 | AUC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SVM-C5 | Full 11 | Max Recall | 0.2637 | 77.68% | 57.37% | 66.00% | 84.64% |
| SVM-C5 | Full 11 | Balance F1 | 0.4053 | 68.96% | 63.23% | 65.97% | 84.64% |
| SVM-C2 | Full 11 | Max Recall | 0.2568 | 77.18% | 58.15% | 66.33% | 84.00% |
| SVM-C2 | Full 11 | Balance F1 | 0.2867 | 74.66% | 59.10% | 65.97% | 84.00% |
| SVM-C4 | Full 11 | Max Recall | 0.2586 | 75.84% | 58.32% | 65.94% | 83.65% |
| SVM-C4 | Full 11 | Balance F1 | 0.2864 | 73.99% | 59.68% | 66.07% | 83.65% |
| SVM-C3 | MI 5 | Max Recall | 0.4216 | 71.48% | 53.12% | 60.94% | 80.99% |
| SVM-C3 | MI 5 | Balance F1 | 0.4438 | 69.63% | 54.39% | 61.07% | 80.99% |
| SVM-C5 | MI 5 | Max Recall | 0.3694 | 71.48% | 52.85% | 60.77% | 80.24% |
| SVM-C5 | MI 5 | Balance F1 | 0.3969 | 69.63% | 54.25% | 60.98% | 80.24% |
| SVM-C4 | MI 5 | Max Recall | 0.2408 | 79.53% | 50.91% | 62.08% | 79.09% |
| SVM-C4 | MI 5 | Balance F1 | 0.3899 | 70.13% | 55.22% | 61.79% | 79.09% |

### 3. Random Forest

| Config | Feature Set | Strategy | Threshold | Recall | Precision | F1 | AUC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RF-C2 | Full 11 | Max Recall | 0.4833 | 76.34% | 57.16% | 65.37% | 84.71% |
| RF-C2 | Full 11 | Balance F1 | 0.5353 | 69.80% | 61.18% | 65.20% | 84.71% |
| RF-C3 | Full 11 | Max Recall | 0.4404 | 77.68% | 54.79% | 64.26% | 84.50% |
| RF-C3 | Full 11 | Balance F1 | 0.5145 | 68.46% | 61.54% | 64.81% | 84.50% |
| RF-C1 | Full 11 | Max Recall | 0.4712 | 81.21% | 54.02% | 64.88% | 84.31% |
| RF-C1 | Full 11 | Balance F1 | 0.5267 | 73.99% | 58.72% | 65.48% | 84.31% |
| RF-C4 | MI 5 | Max Recall | 0.4679 | 71.98% | 51.94% | 60.34% | 80.29% |
| RF-C5 | MI 5 | Max Recall | 0.3672 | 75.17% | 48.33% | 58.83% | 78.49% |
| RF-C2 | MI 5 | Max Recall | 0.4713 | 72.48% | 51.61% | 60.29% | 80.75% |

### 4. XGBoost

| Config | Feature Set | Strategy | Threshold | Recall | Precision | F1 | AUC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| XGB-C4 | Full 11 | Max Recall | 0.2069 | 79.36% | 49.84% | 61.23% | 81.36% |
| XGB-C4 | Full 11 | Balance F1 | 0.2783 | 72.15% | 53.82% | 61.65% | 81.36% |
| XGB-C5 | Full 11 | Max Recall | 0.1929 | 81.71% | 48.17% | 60.61% | 82.26% |
| XGB-C5 | Full 11 | Balance F1 | 0.3462 | 68.62% | 59.71% | 63.86% | 82.26% |
| XGB-C1 | Full 11 | Max Recall | 0.2880 | 78.69% | 55.44% | 65.05% | 84.85% |
| XGB-C1 | Full 11 | Balance F1 | 0.3487 | 72.82% | 60.19% | 65.91% | 84.85% |
| XGB-C5 | MI 5 | Max Recall | 0.2228 | 79.70% | 45.67% | 58.07% | 77.14% |
| XGB-C4 | MI 5 | Max Recall | 0.2803 | 73.66% | 48.89% | 58.77% | 77.23% |
| XGB-C2 | MI 5 | Max Recall | 0.3118 | 73.83% | 50.52% | 59.99% | 79.71% |

### 5. Decision Tree

| Config | Feature Set | Strategy | Threshold | Recall | Precision | F1 | AUC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DT-C1 | Full 11 | Max Recall | 0.2911 | **92.45%** | 44.65% | 60.22% | 82.03% |
| DT-C1 | Full 11 | Balance F1 | 0.3987 | 64.09% | 62.93% | 63.51% | 82.03% |
| DT-C2 | Full 11 | Max Recall | 0.2812 | 78.52% | 52.82% | 63.16% | 82.88% |
| DT-C2 | Full 11 | Balance F1 | 0.4513 | 69.46% | 60.26% | 64.54% | 82.88% |
| DT-C3 | Full 11 | Max Recall | 0.2812 | 79.36% | 52.61% | 63.28% | 82.93% |
| DT-C3 | Full 11 | Balance F1 | 0.3571 | 70.64% | 59.89% | 64.82% | 82.93% |
| DT-C1 | MI 5 | Max Recall | 0.2828 | 90.10% | 44.79% | 59.83% | 80.05% |
| DT-C1 | MI 5 | Balance F1 | 0.4956 | 65.10% | 57.74% | 61.20% | 80.05% |
| DT-C5 | MI 5 | Max Recall | 0.3636 | 75.67% | 44.57% | 56.09% | 73.70% |
| DT-C4 | MI 5 | Max Recall | 0.4167 | 77.85% | 46.63% | 58.33% | 76.99% |

---

## 🔑 Kết Luận

| Tiêu chí | Mô hình tốt nhất | Kết quả nổi bật |
|:---|:---|:---|
| **Recall cao nhất** | Decision Tree (DT-C1) | Recall = **92.45%** (Full 11) |
| **F1-Score cao nhất** | Logistic Regression (LR-C3) | F1 = **67.23%** (Full 11, Balance) |
| **AUC cao nhất** | Logistic Regression (LR-C5) | AUC = **85.08%** |
| **Cân bằng Recall + Precision** | Random Forest (RF-C1) | Recall=81.21%, Prec=54.02%, F1=64.88% |
| **Ổn định nhất trên MI 5** | Logistic Regression | F1 chỉ giảm ~5% khi giảm từ 11→5 features |

> [!TIP]
> File kết quả tổng hợp CSV: [all_models_combined.csv](file:///e:/kpdl_main/models/results/all_models_combined.csv)
