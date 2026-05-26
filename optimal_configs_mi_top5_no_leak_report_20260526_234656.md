# Bao cao thuc nghiem MI Top 5 khong ro ri du lieu

- Ngay chay: 2026-05-26 23:52:47
- MI duoc tinh chi tren train goc truoc SMOTE.
- Validation chi dung de chon threshold; Test chi dung de danh gia cuoi.
- Nguong MI: > 0.01; so feature chon: 5.

## Features duoc chon

| Hang | Feature | MI Score |
| :---: | :--- | ---: |
| 1 | Hypertension | 0.082816 |
| 2 | Age | 0.069118 |
| 3 | Heart_Disease | 0.036362 |
| 4 | Avg_Glucose | 0.028300 |
| 5 | Diabetes | 0.027725 |

## Tap du lieu

- Train sau SMOTE: 7374 mau
- Validation: 1000 mau
- Test: 2000 mau
- Features: Hypertension, Age, Heart_Disease, Avg_Glucose, Diabetes

## Tham chieu cau hinh

| Thuat toan | Config | Cau hinh cu the |
| :--- | :---: | :--- |
| Logistic Regression | LR-C1 | C=0.01, penalty=l2, class_weight=balanced, solver=lbfgs, max_iter=1000, random_state=42 |
| Logistic Regression | LR-C2 | C=0.1, penalty=l2, class_weight=balanced, solver=lbfgs, max_iter=1000, random_state=42 |
| Logistic Regression | LR-C3 | C=1.0, penalty=l2, class_weight=balanced, solver=lbfgs, max_iter=1000, random_state=42 |
| Logistic Regression | LR-C4 | C=10.0, penalty=l2, class_weight=balanced, solver=lbfgs, max_iter=1000, random_state=42 |
| Logistic Regression | LR-C5 | C=100.0, penalty=l2, class_weight=balanced, solver=lbfgs, max_iter=1000, random_state=42 |
| Random Forest | RF-C1 | max_depth=4, n_estimators=50, class_weight=balanced, random_state=42, n_jobs=-1 |
| Random Forest | RF-C2 | max_depth=6, n_estimators=100, class_weight=balanced, random_state=42, n_jobs=-1 |
| Random Forest | RF-C3 | max_depth=8, n_estimators=200, class_weight=balanced, random_state=42, n_jobs=-1 |
| Random Forest | RF-C4 | max_depth=8, n_estimators=300, min_samples_split=10, class_weight=balanced, random_state=42, n_jobs=-1 |
| Random Forest | RF-C5 | max_depth=12, n_estimators=500, class_weight=balanced, random_state=42, n_jobs=-1 |
| XGBoost | XGB-C1 | max_depth=3, learning_rate=0.1, n_estimators=100, eval_metric=logloss, random_state=42, n_jobs=-1 |
| XGBoost | XGB-C2 | max_depth=5, learning_rate=0.05, n_estimators=200, eval_metric=logloss, random_state=42, n_jobs=-1 |
| XGBoost | XGB-C3 | max_depth=4, learning_rate=0.1, n_estimators=150, eval_metric=logloss, random_state=42, n_jobs=-1 |
| XGBoost | XGB-C4 | max_depth=6, learning_rate=0.1, n_estimators=300, eval_metric=logloss, random_state=42, n_jobs=-1 |
| XGBoost | XGB-C5 | max_depth=10, learning_rate=0.01, n_estimators=500, eval_metric=logloss, random_state=42, n_jobs=-1 |
| Decision Tree | DT-C1 | max_depth=3, min_samples_split=2, class_weight=balanced, random_state=42 |
| Decision Tree | DT-C2 | max_depth=5, min_samples_split=5, class_weight=balanced, random_state=42 |
| Decision Tree | DT-C3 | max_depth=6, min_samples_leaf=20, class_weight=balanced, random_state=42 |
| Decision Tree | DT-C4 | max_depth=8, min_samples_split=10, class_weight=balanced, random_state=42 |
| Decision Tree | DT-C5 | max_depth=12, min_samples_split=20, class_weight=balanced, random_state=42 |
| SVM (RBF) | SVM-C1 | C=0.1, kernel=rbf, gamma=scale, class_weight=balanced, probability=True, random_state=42, cache_size=1000 |
| SVM (RBF) | SVM-C2 | C=1.0, kernel=rbf, gamma=scale, class_weight=balanced, probability=True, random_state=42, cache_size=1000 |
| SVM (RBF) | SVM-C3 | C=10.0, kernel=rbf, gamma=0.01, class_weight=balanced, probability=True, random_state=42, cache_size=1000 |
| SVM (RBF) | SVM-C4 | C=10.0, kernel=rbf, gamma=0.1, class_weight=balanced, probability=True, random_state=42, cache_size=1000 |
| SVM (RBF) | SVM-C5 | C=100.0, kernel=rbf, gamma=0.01, class_weight=balanced, probability=True, random_state=42, cache_size=1000 |

## Bang tong hop tat ca cau hinh

| Thuat toan | Thu hang | Config | Threshold | Recall | Precision | F1 | AUC | Trang thai |
| :--- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Logistic Regression | Top 1 | LR-C4 | 0.4480 | 82.55% | 54.19% | 65.43% | 84.63% | DAT >= 80% |
|  | Top 2 | LR-C5 | 0.4480 | 82.55% | 54.19% | 65.43% | 84.63% | DAT >= 80% |
|  | Top 3 | LR-C2 | 0.4560 | 82.21% | 54.26% | 65.38% | 84.57% | DAT >= 80% |
|  | Top 4 | LR-C3 | 0.4425 | 82.55% | 53.59% | 64.99% | 84.63% | DAT >= 80% |
|  | Top 5 | LR-C1 | 0.4660 | 80.87% | 53.03% | 64.05% | 83.95% | DAT >= 80% |
| Random Forest | Top 1 | RF-C1 | 0.4380 | 81.38% | 54.25% | 65.10% | 84.12% | DAT >= 80% |
|  | Top 2 | RF-C5 | 0.3240 | 80.54% | 49.03% | 60.95% | 80.61% | DAT >= 80% |
|  | Top 3 | RF-C2 | 0.4235 | 78.69% | 54.28% | 64.25% | 83.80% | CHUA DAT |
|  | Top 4 | RF-C3 | 0.3920 | 78.86% | 53.05% | 63.43% | 82.77% | CHUA DAT |
|  | Top 5 | RF-C4 | 0.4045 | 77.68% | 53.28% | 63.21% | 82.90% | CHUA DAT |
| XGBoost | Top 1 | XGB-C1 | 0.2610 | 81.04% | 53.85% | 64.70% | 84.11% | DAT >= 80% |
|  | Top 2 | XGB-C5 | 0.2035 | 82.38% | 47.17% | 59.99% | 80.48% | DAT >= 80% |
|  | Top 3 | XGB-C4 | 0.2050 | 80.70% | 47.62% | 59.90% | 80.47% | DAT >= 80% |
|  | Top 4 | XGB-C2 | 0.2620 | 79.19% | 52.68% | 63.27% | 82.82% | CHUA DAT |
|  | Top 5 | XGB-C3 | 0.2510 | 79.36% | 51.75% | 62.65% | 83.17% | CHUA DAT |
| Decision Tree | Top 1 | DT-C3 | 0.4315 | 80.20% | 53.29% | 64.03% | 82.68% | DAT >= 80% |
|  | Top 2 | DT-C2 | 0.3335 | 81.71% | 50.41% | 62.36% | 83.18% | DAT >= 80% |
|  | Top 3 | DT-C1 | 0.2880 | 92.45% | 44.65% | 60.22% | 82.03% | DAT >= 80% |
|  | Top 4 | DT-C4 | 0.3925 | 77.85% | 52.02% | 62.37% | 79.22% | CHUA DAT |
|  | Top 5 | DT-C5 | 0.2860 | 75.17% | 44.53% | 55.93% | 73.12% | CHUA DAT |
| SVM (RBF) | Top 1 | SVM-C3 | 0.2640 | 81.21% | 55.38% | 65.85% | 84.51% | DAT >= 80% |
|  | Top 2 | SVM-C2 | 0.1560 | 81.88% | 54.59% | 65.50% | 82.63% | DAT >= 80% |
|  | Top 3 | SVM-C5 | 0.2155 | 80.54% | 54.98% | 65.35% | 83.98% | DAT >= 80% |
|  | Top 4 | SVM-C4 | 0.1665 | 82.55% | 54.07% | 65.34% | 83.03% | DAT >= 80% |
|  | Top 5 | SVM-C1 | 0.1700 | 82.89% | 52.16% | 64.03% | 83.35% | DAT >= 80% |

## Cau hinh tot nhat moi thuat toan

| Thuat toan | Config | Threshold | Recall | Precision | F1 | AUC |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | LR-C4 | 0.4480 | 82.55% | 54.19% | 65.43% | 84.63% |
| Random Forest | RF-C1 | 0.4380 | 81.38% | 54.25% | 65.10% | 84.12% |
| XGBoost | XGB-C1 | 0.2610 | 81.04% | 53.85% | 64.70% | 84.11% |
| Decision Tree | DT-C3 | 0.4315 | 80.20% | 53.29% | 64.03% | 82.68% |
| SVM (RBF) | SVM-C3 | 0.2640 | 81.21% | 55.38% | 65.85% | 84.51% |