# Bao cao thuc nghiem khong SMOTE theo cau hinh so_sanh_full11_vs_mi5

- Ngay chay: 2026-05-27 21:58:08
- Du lieu train giu nguyen phan phoi goc, khong ap dung SMOTE.
- Split duoc tao lai tu `data_processed.csv` voi random_state=42: 70% train, 10% validation, 20% test.
- Cau hinh chay lai: `LR-C4`, `SVM-C5`, `RF-C3`, `XGB-C3`, `DT-C3` lay tu `so_sanh_full11_vs_mi5.md`.
- Threshold duoc chon tren validation theo chien luoc Extreme Recall, target Recall >= 83%.

## Tap du lieu da tao

| Tap | File | So mau | Stroke=0 | Stroke=1 |
| :--- | :--- | ---: | ---: | ---: |
| Train khong SMOTE | `train_no_smote.csv` | 7000 | 4916 | 2084 |
| Validation | `val_no_smote.csv` | 1000 | 702 | 298 |
| Test | `test_no_smote.csv` | 2000 | 1404 | 596 |

## MI Top 5 tinh tren train khong SMOTE

| Hang | Feature | MI Score |
| :---: | :--- | ---: |
| 1 | Hypertension | 0.082816 |
| 2 | Age | 0.069118 |
| 3 | Heart_Disease | 0.036362 |
| 4 | Avg_Glucose | 0.028300 |
| 5 | Diabetes | 0.027725 |

## Ket qua test

| Bo dac trung | Thuat toan | Config | Threshold | Recall | Precision | F1 | AUC | TP | FP | FN | TN |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full 11 | Logistic Regression | LR-C4 | 0.4440 | 80.54% | 54.24% | 64.82% | 85.11% | 480 | 405 | 116 | 999 |
| Full 11 | SVM (RBF) | SVM-C5 | 0.2040 | 80.70% | 54.11% | 64.78% | 84.77% | 481 | 408 | 115 | 996 |
| Full 11 | Random Forest | RF-C3 | 0.4145 | 80.70% | 53.68% | 64.48% | 84.65% | 481 | 415 | 115 | 989 |
| Full 11 | XGBoost | XGB-C3 | 0.2300 | 81.38% | 53.36% | 64.45% | 83.80% | 485 | 424 | 111 | 980 |
| Full 11 | Decision Tree | DT-C3 | 0.3860 | 82.55% | 50.83% | 62.92% | 83.74% | 492 | 476 | 104 | 928 |
| MI Top 5 | Logistic Regression | LR-C4 | 0.4460 | 82.21% | 54.44% | 65.51% | 84.68% | 490 | 410 | 106 | 994 |
| MI Top 5 | SVM (RBF) | SVM-C5 | 0.2030 | 80.70% | 55.03% | 65.44% | 84.13% | 481 | 393 | 115 | 1011 |
| MI Top 5 | Random Forest | RF-C3 | 0.3575 | 82.05% | 50.36% | 62.41% | 82.89% | 489 | 482 | 107 | 922 |
| MI Top 5 | XGBoost | XGB-C3 | 0.2200 | 79.87% | 52.14% | 63.09% | 83.09% | 476 | 437 | 120 | 967 |
| MI Top 5 | Decision Tree | DT-C3 | 0.4315 | 80.20% | 53.29% | 64.03% | 82.93% | 478 | 419 | 118 | 985 |

## So sanh Full 11 va MI Top 5 trong dieu kien khong SMOTE

| Thuat toan | Cau hinh da chay | Recall Full 11 | Recall MI Top 5 | Chenh Recall | F1 Full 11 | F1 MI Top 5 | Chenh F1 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | LR-C4<br>`C=10.0, penalty=l2, class_weight=balanced, solver=lbfgs, max_iter=1000, random_state=42` | 80.54% | 82.21% | 1.68% | 64.82% | 65.51% | 0.69% |
| SVM (RBF) | SVM-C5<br>`C=100.0, kernel=rbf, gamma=0.01, class_weight=balanced, probability=True, random_state=42, cache_size=1000` | 80.70% | 80.70% | 0.00% | 64.78% | 65.44% | 0.66% |
| Random Forest | RF-C3<br>`max_depth=8, n_estimators=200, class_weight=balanced, random_state=42, n_jobs=-1` | 80.70% | 82.05% | 1.34% | 64.48% | 62.41% | -2.06% |
| XGBoost | XGB-C3<br>`max_depth=4, learning_rate=0.1, n_estimators=150, eval_metric=logloss, random_state=42, n_jobs=-1` | 81.38% | 79.87% | -1.51% | 64.45% | 63.09% | -1.36% |
| Decision Tree | DT-C3<br>`max_depth=6, min_samples_leaf=20, class_weight=balanced, random_state=42` | 82.55% | 80.20% | -2.35% | 62.92% | 64.03% | 1.12% |

## Nhan xet nhanh

- F1 cao nhat: Logistic Regression LR-C4 tren MI Top 5 voi F1=65.51%, Recall=82.21%.
- Recall cao nhat: Decision Tree DT-C3 tren Full 11 voi Recall=82.55%, Precision=50.83%.
- Neu uu tien sang loc y khoa, can doc Recall cung voi FP/FN vi bo du lieu khong SMOTE lam lop Stroke=1 rat thua.
