"""
Run selected model configurations from so_sanh_full11_vs_mi5.md without SMOTE.

The script rebuilds the original 70/10/20 split from data_processed.csv,
keeps the training set imbalanced, evaluates the five representative
configurations on both Full 11 and train-only MI Top 5 feature sets, and writes
CSV/Markdown outputs.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_PROCESSED = "data_processed.csv"
TARGET_COL = "Stroke"
TOP_N = 5
MI_THRESHOLD = 0.01
CONTINUOUS_BINS = 10
TARGET_RECALL = 0.83

OUTPUTS = {
    "train_full": Path("train_no_smote.csv"),
    "val_full": Path("val_no_smote.csv"),
    "test_full": Path("test_no_smote.csv"),
    "mi_scores": Path("mi_top5_no_smote_train_only_scores.csv"),
    "train_mi": Path("train_no_smote_mi_top5.csv"),
    "val_mi": Path("val_no_smote_mi_top5.csv"),
    "test_mi": Path("test_no_smote_mi_top5.csv"),
    "results": Path("no_smote_selected_configs_results.csv"),
    "report": Path("bao_cao_khong_smote_theo_so_sanh.md"),
}


def p(message):
    print(message, flush=True)


def split_without_smote():
    df = pd.read_csv(DATA_PROCESSED)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    x = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    x_temp_train, x_test, y_temp_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp_train,
        y_temp_train,
        test_size=0.125,
        random_state=42,
        stratify=y_temp_train,
    )

    train_df = pd.concat([x_train, y_train], axis=1)
    val_df = pd.concat([x_val, y_val], axis=1)
    test_df = pd.concat([x_test, y_test], axis=1)
    return train_df, val_df, test_df


def discretize_feature(series):
    if series.nunique(dropna=True) <= 10:
        return series.fillna("missing").astype(str)

    ranked = series.rank(method="first")
    bins = min(CONTINUOUS_BINS, series.nunique(dropna=True))
    return pd.qcut(ranked, q=bins, duplicates="drop").astype(str)


def mutual_information_score(feature, target):
    data = pd.DataFrame({"feature": feature, "target": target}).dropna()
    joint = pd.crosstab(data["feature"], data["target"], normalize=True)
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    score = 0.0
    for feature_value in joint.index:
        for target_value in joint.columns:
            pxy = joint.loc[feature_value, target_value]
            if pxy > 0:
                score += pxy * np.log(
                    pxy / (px.loc[feature_value] * py.loc[target_value])
                )
    return float(score)


def select_mi_features(train_df):
    x_train = train_df.drop(TARGET_COL, axis=1)
    y_train = train_df[TARGET_COL]
    scores = [
        mutual_information_score(discretize_feature(x_train[col]), y_train)
        for col in x_train.columns
    ]
    score_df = (
        pd.DataFrame({"Feature": x_train.columns, "MI_Score": scores})
        .sort_values("MI_Score", ascending=False)
        .reset_index(drop=True)
    )
    selected = score_df.loc[score_df["MI_Score"] > MI_THRESHOLD, "Feature"].head(TOP_N)
    selected_features = selected.tolist()
    if len(selected_features) < TOP_N:
        raise ValueError(
            f"Only {len(selected_features)} features passed MI > {MI_THRESHOLD}; "
            f"need {TOP_N}."
        )
    return score_df, selected_features


def subset_features(df, selected_features):
    return df[selected_features + [TARGET_COL]].copy()


def selected_configs_from_comparison_report():
    return [
        (
            "Logistic Regression",
            "LR-C4",
            LogisticRegression,
            {
                "C": 10.0,
                "penalty": "l2",
                "class_weight": "balanced",
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": 42,
            },
        ),
        (
            "SVM (RBF)",
            "SVM-C5",
            SVC,
            {
                "C": 100.0,
                "kernel": "rbf",
                "gamma": 0.01,
                "class_weight": "balanced",
                "probability": True,
                "random_state": 42,
                "cache_size": 1000,
            },
        ),
        (
            "Random Forest",
            "RF-C3",
            RandomForestClassifier,
            {
                "max_depth": 8,
                "n_estimators": 200,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            },
        ),
        (
            "XGBoost",
            "XGB-C3",
            XGBClassifier,
            {
                "max_depth": 4,
                "learning_rate": 0.1,
                "n_estimators": 150,
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": -1,
            },
        ),
        (
            "Decision Tree",
            "DT-C3",
            DecisionTreeClassifier,
            {
                "max_depth": 6,
                "min_samples_leaf": 20,
                "class_weight": "balanced",
                "random_state": 42,
            },
        ),
    ]


def get_proba(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    scores = model.decision_function(x)
    return 1 / (1 + np.exp(-scores))


def find_extreme_recall_threshold(y_true, y_proba, target_recall=TARGET_RECALL):
    best_threshold = 0.5
    best_f1 = -1

    for threshold in np.arange(0.01, 0.99, 0.0005):
        y_pred = (y_proba >= threshold).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if rec >= target_recall and f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    if best_f1 == -1:
        best_score = -1
        for threshold in np.arange(0.01, 0.99, 0.0005):
            y_pred = (y_proba >= threshold).astype(int)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            score = rec * 0.7 + f1 * 0.3
            if score > best_score:
                best_score = score
                best_threshold = threshold

    return float(best_threshold)


def evaluate_predictions(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def format_params(params):
    return ", ".join(f"{key}={value}" for key, value in params.items())


def run_one_dataset(dataset_name, train_df, val_df, test_df):
    x_train = train_df.drop(TARGET_COL, axis=1)
    y_train = train_df[TARGET_COL]
    x_val = val_df.drop(TARGET_COL, axis=1)
    y_val = val_df[TARGET_COL]
    x_test = test_df.drop(TARGET_COL, axis=1)
    y_test = test_df[TARGET_COL]

    rows = []
    for algo_name, config_name, model_class, params in selected_configs_from_comparison_report():
        p(f"[{dataset_name}] {algo_name} {config_name}: training...")
        start = time.time()
        model = model_class(**params)
        model.fit(x_train, y_train)
        train_time = time.time() - start

        val_proba = get_proba(model, x_val)
        test_proba = get_proba(model, x_test)
        threshold = find_extreme_recall_threshold(y_val, val_proba)
        metrics = evaluate_predictions(y_test, test_proba, threshold)

        row = {
            "dataset": dataset_name,
            "algorithm": algo_name,
            "config": config_name,
            "features": x_train.shape[1],
            "params": str(params),
            "param_summary": format_params(params),
            "train_time_sec": train_time,
            **metrics,
        }
        rows.append(row)
        p(
            f"  threshold={threshold:.4f} | recall={metrics['recall'] * 100:.2f}% | "
            f"precision={metrics['precision'] * 100:.2f}% | "
            f"f1={metrics['f1'] * 100:.2f}% | auc={metrics['auc'] * 100:.2f}%"
        )
    return rows


def pct(value):
    return f"{value * 100:.2f}%"


def make_report(
    train_df,
    val_df,
    test_df,
    score_df,
    selected_features,
    results_df,
):
    lines = []
    lines.append("# Bao cao thuc nghiem khong SMOTE theo cau hinh so_sanh_full11_vs_mi5")
    lines.append("")
    lines.append(f"- Ngay chay: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- Du lieu train giu nguyen phan phoi goc, khong ap dung SMOTE.")
    lines.append("- Split duoc tao lai tu `data_processed.csv` voi random_state=42: 70% train, 10% validation, 20% test.")
    lines.append("- Cau hinh chay lai: `LR-C4`, `SVM-C5`, `RF-C3`, `XGB-C3`, `DT-C3` lay tu `so_sanh_full11_vs_mi5.md`.")
    lines.append(f"- Threshold duoc chon tren validation theo chien luoc Extreme Recall, target Recall >= {TARGET_RECALL * 100:.0f}%.")
    lines.append("")
    lines.append("## Tap du lieu da tao")
    lines.append("")
    lines.append("| Tap | File | So mau | Stroke=0 | Stroke=1 |")
    lines.append("| :--- | :--- | ---: | ---: | ---: |")
    for label, path, df in [
        ("Train khong SMOTE", OUTPUTS["train_full"], train_df),
        ("Validation", OUTPUTS["val_full"], val_df),
        ("Test", OUTPUTS["test_full"], test_df),
    ]:
        counts = df[TARGET_COL].value_counts().to_dict()
        lines.append(
            f"| {label} | `{path}` | {len(df)} | {counts.get(0, 0)} | {counts.get(1, 0)} |"
        )
    lines.append("")
    lines.append("## MI Top 5 tinh tren train khong SMOTE")
    lines.append("")
    lines.append("| Hang | Feature | MI Score |")
    lines.append("| :---: | :--- | ---: |")
    for i, feature in enumerate(selected_features, 1):
        score = score_df.loc[score_df["Feature"] == feature, "MI_Score"].iloc[0]
        lines.append(f"| {i} | {feature} | {score:.6f} |")
    lines.append("")
    lines.append("## Ket qua test")
    lines.append("")
    lines.append("| Bo dac trung | Thuat toan | Config | Threshold | Recall | Precision | F1 | AUC | TP | FP | FN | TN |")
    lines.append("| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in results_df.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['algorithm']} | {row['config']} | "
            f"{row['threshold']:.4f} | {pct(row['recall'])} | {pct(row['precision'])} | "
            f"{pct(row['f1'])} | {pct(row['auc'])} | {row['tp']} | {row['fp']} | "
            f"{row['fn']} | {row['tn']} |"
        )
    lines.append("")
    lines.append("## So sanh Full 11 va MI Top 5 trong dieu kien khong SMOTE")
    lines.append("")
    lines.append("| Thuat toan | Cau hinh da chay | Recall Full 11 | Recall MI Top 5 | Chenh Recall | F1 Full 11 | F1 MI Top 5 | Chenh F1 |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for algorithm in results_df["algorithm"].unique():
        full_row = results_df[
            (results_df["dataset"] == "Full 11") & (results_df["algorithm"] == algorithm)
        ].iloc[0]
        mi_row = results_df[
            (results_df["dataset"] == "MI Top 5") & (results_df["algorithm"] == algorithm)
        ].iloc[0]
        config_cell = f"{full_row['config']}<br>`{full_row['param_summary']}`"
        lines.append(
            f"| {algorithm} | {config_cell} | {pct(full_row['recall'])} | "
            f"{pct(mi_row['recall'])} | {pct(mi_row['recall'] - full_row['recall'])} | "
            f"{pct(full_row['f1'])} | {pct(mi_row['f1'])} | "
            f"{pct(mi_row['f1'] - full_row['f1'])} |"
        )
    lines.append("")
    lines.append("## Nhan xet nhanh")
    lines.append("")
    best_f1 = results_df.sort_values("f1", ascending=False).iloc[0]
    best_recall = results_df.sort_values("recall", ascending=False).iloc[0]
    lines.append(
        f"- F1 cao nhat: {best_f1['algorithm']} {best_f1['config']} tren {best_f1['dataset']} "
        f"voi F1={pct(best_f1['f1'])}, Recall={pct(best_f1['recall'])}."
    )
    lines.append(
        f"- Recall cao nhat: {best_recall['algorithm']} {best_recall['config']} tren {best_recall['dataset']} "
        f"voi Recall={pct(best_recall['recall'])}, Precision={pct(best_recall['precision'])}."
    )
    lines.append("- Neu uu tien sang loc y khoa, can doc Recall cung voi FP/FN vi bo du lieu khong SMOTE lam lop Stroke=1 rat thua.")
    return "\n".join(lines)


def main():
    p("=" * 80)
    p("TAO DU LIEU KHONG SMOTE VA CHAY CAU HINH TU BAO CAO SO SANH")
    p("=" * 80)

    train_df, val_df, test_df = split_without_smote()
    train_df.to_csv(OUTPUTS["train_full"], index=False)
    val_df.to_csv(OUTPUTS["val_full"], index=False)
    test_df.to_csv(OUTPUTS["test_full"], index=False)
    p("Saved no-SMOTE full datasets.")

    score_df, selected_features = select_mi_features(train_df)
    score_df.to_csv(OUTPUTS["mi_scores"], index=False)
    train_mi = subset_features(train_df, selected_features)
    val_mi = subset_features(val_df, selected_features)
    test_mi = subset_features(test_df, selected_features)
    train_mi.to_csv(OUTPUTS["train_mi"], index=False)
    val_mi.to_csv(OUTPUTS["val_mi"], index=False)
    test_mi.to_csv(OUTPUTS["test_mi"], index=False)
    p(f"Selected MI features: {selected_features}")

    rows = []
    rows.extend(run_one_dataset("Full 11", train_df, val_df, test_df))
    rows.extend(run_one_dataset("MI Top 5", train_mi, val_mi, test_mi))
    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUTS["results"], index=False)

    report = make_report(
        train_df,
        val_df,
        test_df,
        score_df,
        selected_features,
        results_df,
    )
    OUTPUTS["report"].write_text(report, encoding="utf-8")

    p("\nOutputs:")
    for path in OUTPUTS.values():
        p(f"  - {path}")
    p("DONE.")


if __name__ == "__main__":
    main()
