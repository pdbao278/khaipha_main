"""
Create the algorithm metric comparison chart for the full 11-feature dataset.

Output:
- visuals/full11_algorithm_metric_comparison.png

If the output already exists, a timestamp is appended to avoid overwriting.
"""

from pathlib import Path
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPORT_FILE = Path("report_top5_full11.txt")
OUTPUT_DIR = Path("visuals")
OUTPUT_FILE = "full11_algorithm_metric_comparison.png"


BEST_ROW_RE = re.compile(
    r"^(Logistic Regression|Random Forest|XGBoost|Decision Tree|SVM \(RBF\))\s+"
    r"([A-Z]+-C\d+)\s+"
    r"([\d.]+)%\s+"
    r"([\d.]+)%\s+"
    r"([\d.]+)%\s+"
    r"([\d.]+)%\s+"
    r"([\d.]+)"
)


def safe_path(path):
    if not path.exists():
        return path
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def parse_best_table(report_path):
    if not report_path.exists():
        raise FileNotFoundError(f"Khong tim thay file bao cao: {report_path}")

    rows = []
    in_best_section = False
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if "BANG TONG HOP CAU HINH TOT NHAT MOI THUAT TOAN" in line:
            in_best_section = True
            continue
        if not in_best_section:
            continue

        match = BEST_ROW_RE.match(line.strip())
        if not match:
            continue

        algo, config, recall, precision, f1, auc, threshold = match.groups()
        rows.append(
            {
                "Algorithm": algo,
                "Config": config,
                "Recall": float(recall),
                "Precision": float(precision),
                "F1": float(f1),
                "AUC": float(auc),
                "Threshold": float(threshold),
            }
        )

    if len(rows) != 5:
        raise ValueError(f"Bang best config full 11 khong du 5 dong, hien co {len(rows)}.")
    return pd.DataFrame(rows)


def add_bar_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_algorithm_metric_comparison(df, output_path):
    metrics = ["Recall", "Precision", "F1", "AUC"]
    x = np.arange(len(df))
    width = 0.18
    colors = ["#2f80ed", "#27ae60", "#f2994a", "#9b51e0"]

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, df[metric], width, label=metric, color=colors[i])
        add_bar_labels(ax, bars)

    labels = df["Algorithm"] + "\n" + df["Config"]
    ax.set_title("So sanh cac thuat toan - Full 11 features", fontsize=15, weight="bold")
    ax.set_ylabel("Ty le (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = parse_best_table(REPORT_FILE)
    output_path = safe_path(OUTPUT_DIR / OUTPUT_FILE)
    plot_algorithm_metric_comparison(df, output_path)

    print(f"Da doc bao cao: {REPORT_FILE}")
    print("Best config full 11:")
    print(df[["Algorithm", "Config", "Recall", "Precision", "F1", "AUC"]].to_string(index=False))
    print(f"Da tao bieu do: {output_path}")


if __name__ == "__main__":
    main()
