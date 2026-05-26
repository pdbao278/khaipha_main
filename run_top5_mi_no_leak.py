"""
Run top-5 MI feature experiments without data leakage.

This script selects features using only the original training split before
SMOTE, then evaluates the same five configurations per algorithm used in the
full-11-feature experiment. It never overwrites existing output files.
"""

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_PROCESSED = "data_processed.csv"
TRAIN_RESAMPLED = "train_resampled.csv"
VAL_FILE = "val.csv"
TEST_FILE = "test.csv"
TARGET_COL = "Stroke"
MI_THRESHOLD = 0.01
TOP_N = 5
CONTINUOUS_BINS = 10

BASE_OUTPUTS = {
    "scores": "mi_top5_train_only_scores.csv",
    "train": "train_mi_top5.csv",
    "val": "val_mi_top5.csv",
    "test": "test_mi_top5.csv",
    "text_report": "report_top5_mi_no_leak.txt",
    "md_report": "optimal_configs_mi_top5_no_leak_report.md",
}


def p(message):
    print(message, flush=True)


def safe_output_paths(base_outputs):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    paths = {}
    for key, filename in base_outputs.items():
        path = Path(filename)
        if path.exists():
            path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        paths[key] = path
    return paths


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


def reconstruct_original_train_split():
    df = pd.read_csv(DATA_PROCESSED)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Khong tim thay cot muc tieu: {TARGET_COL}")

    x = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    x_temp_train, _, y_temp_train, _ = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )
    x_train_raw, _, y_train_raw, _ = train_test_split(
        x_temp_train,
        y_temp_train,
        test_size=0.125,
        random_state=42,
        stratify=y_temp_train,
    )
    return x_train_raw, y_train_raw


def select_mi_features_from_train_only(x_train_raw, y_train_raw):
    mi_scores = [
        mutual_information_score(discretize_feature(x_train_raw[col]), y_train_raw)
        for col in x_train_raw.columns
    ]
    score_df = (
        pd.DataFrame({"Feature": x_train_raw.columns, "MI_Score": mi_scores})
        .sort_values("MI_Score", ascending=False)
        .reset_index(drop=True)
    )

    selected = score_df.loc[score_df["MI_Score"] > MI_THRESHOLD, "Feature"].head(TOP_N)
    selected_features = selected.tolist()
    if len(selected_features) < TOP_N:
        raise ValueError(
            f"Chi co {len(selected_features)} dac trung co MI > {MI_THRESHOLD}; "
            f"khong du top {TOP_N}."
        )
    return score_df, selected_features


def load_and_subset_model_data(selected_features):
    train_df = pd.read_csv(TRAIN_RESAMPLED)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)
    required = selected_features + [TARGET_COL]

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Tap {name} thieu cot: {missing}")

    train_mi = train_df[required].copy()
    val_mi = val_df[required].copy()
    test_mi = test_df[required].copy()

    assert train_mi.shape[1] == TOP_N + 1
    assert val_mi.shape[1] == TOP_N + 1
    assert test_mi.shape[1] == TOP_N + 1
    assert len(train_mi) == len(train_df)
    assert len(val_mi) == len(val_df)
    assert len(test_mi) == len(test_df)
    assert train_mi[TARGET_COL].value_counts().to_dict() == train_df[
        TARGET_COL
    ].value_counts().to_dict()
    assert val_mi[TARGET_COL].value_counts().to_dict() == val_df[
        TARGET_COL
    ].value_counts().to_dict()
    assert test_mi[TARGET_COL].value_counts().to_dict() == test_df[
        TARGET_COL
    ].value_counts().to_dict()

    return train_mi, val_mi, test_mi


def find_extreme_recall_threshold(y_true, y_proba, target_recall=0.83):
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


def get_proba(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    scores = model.decision_function(x)
    return 1 / (1 + np.exp(-scores))


def evaluate(model, x, y, threshold):
    y_proba = get_proba(model, x)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "recall": recall_score(y, y_pred, zero_division=0),
        "precision": precision_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "auc": roc_auc_score(y, y_proba),
        "threshold": threshold,
    }


def model_configs():
    lr_configs = [
        {"name": "LR-C1", "params": {"C": 0.01, "penalty": "l2", "class_weight": "balanced", "solver": "lbfgs", "max_iter": 1000, "random_state": 42}},
        {"name": "LR-C2", "params": {"C": 0.1, "penalty": "l2", "class_weight": "balanced", "solver": "lbfgs", "max_iter": 1000, "random_state": 42}},
        {"name": "LR-C3", "params": {"C": 1.0, "penalty": "l2", "class_weight": "balanced", "solver": "lbfgs", "max_iter": 1000, "random_state": 42}},
        {"name": "LR-C4", "params": {"C": 10.0, "penalty": "l2", "class_weight": "balanced", "solver": "lbfgs", "max_iter": 1000, "random_state": 42}},
        {"name": "LR-C5", "params": {"C": 100.0, "penalty": "l2", "class_weight": "balanced", "solver": "lbfgs", "max_iter": 1000, "random_state": 42}},
    ]
    svm_configs = [
        {"name": "SVM-C1", "params": {"C": 0.1, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced", "probability": True, "random_state": 42, "cache_size": 1000}},
        {"name": "SVM-C2", "params": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced", "probability": True, "random_state": 42, "cache_size": 1000}},
        {"name": "SVM-C3", "params": {"C": 10.0, "kernel": "rbf", "gamma": 0.01, "class_weight": "balanced", "probability": True, "random_state": 42, "cache_size": 1000}},
        {"name": "SVM-C4", "params": {"C": 10.0, "kernel": "rbf", "gamma": 0.1, "class_weight": "balanced", "probability": True, "random_state": 42, "cache_size": 1000}},
        {"name": "SVM-C5", "params": {"C": 100.0, "kernel": "rbf", "gamma": 0.01, "class_weight": "balanced", "probability": True, "random_state": 42, "cache_size": 1000}},
    ]
    rf_configs = [
        {"name": "RF-C1", "params": {"max_depth": 4, "n_estimators": 50, "class_weight": "balanced", "random_state": 42, "n_jobs": -1}},
        {"name": "RF-C2", "params": {"max_depth": 6, "n_estimators": 100, "class_weight": "balanced", "random_state": 42, "n_jobs": -1}},
        {"name": "RF-C3", "params": {"max_depth": 8, "n_estimators": 200, "class_weight": "balanced", "random_state": 42, "n_jobs": -1}},
        {"name": "RF-C4", "params": {"max_depth": 8, "n_estimators": 300, "min_samples_split": 10, "class_weight": "balanced", "random_state": 42, "n_jobs": -1}},
        {"name": "RF-C5", "params": {"max_depth": 12, "n_estimators": 500, "class_weight": "balanced", "random_state": 42, "n_jobs": -1}},
    ]
    xgb_configs = [
        {"name": "XGB-C1", "params": {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100, "eval_metric": "logloss", "random_state": 42, "n_jobs": -1}},
        {"name": "XGB-C2", "params": {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 200, "eval_metric": "logloss", "random_state": 42, "n_jobs": -1}},
        {"name": "XGB-C3", "params": {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 150, "eval_metric": "logloss", "random_state": 42, "n_jobs": -1}},
        {"name": "XGB-C4", "params": {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 300, "eval_metric": "logloss", "random_state": 42, "n_jobs": -1}},
        {"name": "XGB-C5", "params": {"max_depth": 10, "learning_rate": 0.01, "n_estimators": 500, "eval_metric": "logloss", "random_state": 42, "n_jobs": -1}},
    ]
    dt_configs = [
        {"name": "DT-C1", "params": {"max_depth": 3, "min_samples_split": 2, "class_weight": "balanced", "random_state": 42}},
        {"name": "DT-C2", "params": {"max_depth": 5, "min_samples_split": 5, "class_weight": "balanced", "random_state": 42}},
        {"name": "DT-C3", "params": {"max_depth": 6, "min_samples_leaf": 20, "class_weight": "balanced", "random_state": 42}},
        {"name": "DT-C4", "params": {"max_depth": 8, "min_samples_split": 10, "class_weight": "balanced", "random_state": 42}},
        {"name": "DT-C5", "params": {"max_depth": 12, "min_samples_split": 20, "class_weight": "balanced", "random_state": 42}},
    ]
    return [
        ("Logistic Regression", LogisticRegression, lr_configs),
        ("Random Forest", RandomForestClassifier, rf_configs),
        ("XGBoost", XGBClassifier, xgb_configs),
        ("Decision Tree", DecisionTreeClassifier, dt_configs),
        ("SVM (RBF)", SVC, svm_configs),
    ]


def format_params(params):
    parts = []
    for key, value in params.items():
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def run_experiments(train_mi, val_mi, test_mi):
    x_train = train_mi.drop(TARGET_COL, axis=1)
    y_train = train_mi[TARGET_COL]
    x_val = val_mi.drop(TARGET_COL, axis=1)
    y_val = val_mi[TARGET_COL]
    x_test = test_mi.drop(TARGET_COL, axis=1)
    y_test = test_mi[TARGET_COL]

    all_results = {}
    for algo_name, model_class, configs in model_configs():
        p(f"\n{'=' * 80}")
        p(f"THUAT TOAN: {algo_name}")
        p(f"{'=' * 80}")
        algo_results = []

        for cfg in configs:
            cfg_name = cfg["name"]
            p(f"  [{cfg_name}] Huan luyen...")
            start = time.time()
            model = model_class(**cfg["params"])
            model.fit(x_train, y_train)
            train_time = time.time() - start

            val_proba = get_proba(model, x_val)
            threshold = find_extreme_recall_threshold(
                y_val, val_proba, target_recall=0.83
            )
            result = evaluate(model, x_test, y_test, threshold)
            result["config_name"] = cfg_name
            result["params"] = str(cfg["params"])
            result["train_time"] = train_time
            algo_results.append(result)

            mark = "OK" if result["recall"] >= 0.80 else "LOW"
            p(
                f"    -> Nguong={result['threshold']:.4f} | "
                f"Recall={result['recall'] * 100:.2f}% | "
                f"Prec={result['precision'] * 100:.2f}% | "
                f"F1={result['f1'] * 100:.2f}% | "
                f"AUC={result['auc'] * 100:.2f}% [{mark}]"
            )

        if len(algo_results) != 5:
            raise AssertionError(f"{algo_name} khong chay du 5 cau hinh")

        algo_results.sort(
            key=lambda x: (
                1 if x["recall"] >= 0.80 else 0,
                x["f1"],
                x["recall"],
            ),
            reverse=True,
        )
        all_results[algo_name] = algo_results

    return all_results


def make_text_report(paths, score_df, selected_features, train_mi, val_mi, test_mi, results):
    lines = []
    lines.append("=" * 90)
    lines.append("BAO CAO TOP 5 CAU HINH MOI THUAT TOAN - MI TOP 5 TRAIN-ONLY")
    lines.append(f"Ngay chay: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Chien luoc MI: tinh chi tren train goc truoc SMOTE de tranh leakage")
    lines.append("Chien luoc nguong: Extreme Recall (target Recall >= 83% tren Validation)")
    lines.append("=" * 90)
    lines.append("")
    lines.append("OUTPUT:")
    for key, path in paths.items():
        lines.append(f"  - {key}: {path}")
    lines.append("")
    lines.append("FEATURES DUOC CHON:")
    for feature in selected_features:
        score = score_df.loc[score_df["Feature"] == feature, "MI_Score"].iloc[0]
        lines.append(f"  - {feature}: MI={score:.6f}")
    lines.append("")
    lines.append("TAP DU LIEU:")
    lines.append(f"  - Train (sau SMOTE 0.5): {train_mi.shape[0]} mau")
    lines.append(f"  - Validation: {val_mi.shape[0]} mau")
    lines.append(f"  - Test: {test_mi.shape[0]} mau")
    lines.append(f"  - So features: {len(selected_features)}")
    lines.append(f"  - Features: {selected_features}")
    lines.append("")

    for algo_name, algo_results in results.items():
        lines.append("=" * 90)
        lines.append(f"THUAT TOAN: {algo_name}")
        lines.append("=" * 90)
        for i, result in enumerate(algo_results, 1):
            status = "DAT (>= 80%)" if result["recall"] >= 0.80 else "CHUA DAT (< 80%)"
            lines.append(f"\n  --- Top {i}: {result['config_name']} ---")
            lines.append(f"  Sieu tham so: {result['params']}")
            lines.append(f"  Nguong quyet dinh: {result['threshold']:.4f}")
            lines.append(f"  Thoi gian huan luyen: {result['train_time']:.2f}s")
            lines.append("  KET QUA TREN TAP TEST:")
            lines.append(f"    - Recall (Lop 1):    {result['recall'] * 100:.2f}%  [{status}]")
            lines.append(f"    - Precision (Lop 1): {result['precision'] * 100:.2f}%")
            lines.append(f"    - F1-Score (Lop 1):  {result['f1'] * 100:.2f}%")
            lines.append(f"    - AUC-ROC:           {result['auc'] * 100:.2f}%")
        lines.append("")

    lines.append("=" * 90)
    lines.append("BANG TONG HOP CAU HINH TOT NHAT MOI THUAT TOAN (Top 1)")
    lines.append("=" * 90)
    lines.append(
        f"{'Thuat toan':<25} {'Config':<10} {'Recall':>10} {'Prec':>10} "
        f"{'F1':>10} {'AUC':>10} {'Nguong':>8}"
    )
    lines.append("-" * 90)
    for algo_name, algo_results in results.items():
        result = algo_results[0]
        lines.append(
            f"{algo_name:<25} {result['config_name']:<10} "
            f"{result['recall'] * 100:>9.2f}% "
            f"{result['precision'] * 100:>9.2f}% "
            f"{result['f1'] * 100:>9.2f}% "
            f"{result['auc'] * 100:>9.2f}% "
            f"{result['threshold']:>7.4f}"
        )
    return "\n".join(lines)


def make_markdown_report(score_df, selected_features, train_mi, val_mi, test_mi, results):
    lines = []
    lines.append("# Bao cao thuc nghiem MI Top 5 khong ro ri du lieu")
    lines.append("")
    lines.append(f"- Ngay chay: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- MI duoc tinh chi tren train goc truoc SMOTE.")
    lines.append("- Validation chi dung de chon threshold; Test chi dung de danh gia cuoi.")
    lines.append(f"- Nguong MI: > {MI_THRESHOLD}; so feature chon: {TOP_N}.")
    lines.append("")
    lines.append("## Features duoc chon")
    lines.append("")
    lines.append("| Hang | Feature | MI Score |")
    lines.append("| :---: | :--- | ---: |")
    for i, feature in enumerate(selected_features, 1):
        score = score_df.loc[score_df["Feature"] == feature, "MI_Score"].iloc[0]
        lines.append(f"| {i} | {feature} | {score:.6f} |")
    lines.append("")
    lines.append("## Tap du lieu")
    lines.append("")
    lines.append(f"- Train sau SMOTE: {train_mi.shape[0]} mau")
    lines.append(f"- Validation: {val_mi.shape[0]} mau")
    lines.append(f"- Test: {test_mi.shape[0]} mau")
    lines.append(f"- Features: {', '.join(selected_features)}")
    lines.append("")
    lines.append("## Tham chieu cau hinh")
    lines.append("")
    lines.append("| Thuat toan | Config | Cau hinh cu the |")
    lines.append("| :--- | :---: | :--- |")
    for algo_name, _, configs in model_configs():
        for cfg in configs:
            lines.append(
                f"| {algo_name} | {cfg['name']} | {format_params(cfg['params'])} |"
            )
    lines.append("")
    lines.append("## Bang tong hop tat ca cau hinh")
    lines.append("")
    lines.append("| Thuat toan | Thu hang | Config | Threshold | Recall | Precision | F1 | AUC | Trang thai |")
    lines.append("| :--- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | :---: |")
    for algo_name, algo_results in results.items():
        for i, result in enumerate(algo_results, 1):
            status = "DAT >= 80%" if result["recall"] >= 0.80 else "CHUA DAT"
            lines.append(
                f"| {algo_name if i == 1 else ''} | Top {i} | "
                f"{result['config_name']} | {result['threshold']:.4f} | "
                f"{result['recall'] * 100:.2f}% | "
                f"{result['precision'] * 100:.2f}% | "
                f"{result['f1'] * 100:.2f}% | "
                f"{result['auc'] * 100:.2f}% | {status} |"
            )
    lines.append("")
    lines.append("## Cau hinh tot nhat moi thuat toan")
    lines.append("")
    lines.append("| Thuat toan | Config | Threshold | Recall | Precision | F1 | AUC |")
    lines.append("| :--- | :---: | ---: | ---: | ---: | ---: | ---: |")
    for algo_name, algo_results in results.items():
        result = algo_results[0]
        lines.append(
            f"| {algo_name} | {result['config_name']} | {result['threshold']:.4f} | "
            f"{result['recall'] * 100:.2f}% | "
            f"{result['precision'] * 100:.2f}% | "
            f"{result['f1'] * 100:.2f}% | "
            f"{result['auc'] * 100:.2f}% |"
        )
    return "\n".join(lines)


def main():
    paths = safe_output_paths(BASE_OUTPUTS)

    p("=" * 80)
    p("CHON MI TOP 5 TU TRAIN GOC TRUOC SMOTE")
    p("=" * 80)
    x_train_raw, y_train_raw = reconstruct_original_train_split()
    score_df, selected_features = select_mi_features_from_train_only(
        x_train_raw, y_train_raw
    )
    p(f"Features duoc chon: {selected_features}")

    train_mi, val_mi, test_mi = load_and_subset_model_data(selected_features)

    score_df.to_csv(paths["scores"], index=False)
    train_mi.to_csv(paths["train"], index=False)
    val_mi.to_csv(paths["val"], index=False)
    test_mi.to_csv(paths["test"], index=False)

    p("\nDa luu cac tap MI top 5 khong ghi de:")
    for key in ["scores", "train", "val", "test"]:
        p(f"  - {key}: {paths[key]}")

    results = run_experiments(train_mi, val_mi, test_mi)
    text_report = make_text_report(
        paths, score_df, selected_features, train_mi, val_mi, test_mi, results
    )
    md_report = make_markdown_report(
        score_df, selected_features, train_mi, val_mi, test_mi, results
    )

    paths["text_report"].write_text(text_report, encoding="utf-8")
    paths["md_report"].write_text(md_report, encoding="utf-8")

    p("\nDa luu bao cao:")
    p(f"  - {paths['text_report']}")
    p(f"  - {paths['md_report']}")
    p("HOAN TAT.")


if __name__ == "__main__":
    main()
