"""
Create comparison charts from the latest MI top-5 no-leak report.

Outputs:
- visuals/mi_algorithm_metric_comparison.png
- visuals/mi_best_config_comparison.png

If an output already exists, a timestamp is appended to avoid overwriting.
"""

from pathlib import Path
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPORT_PATTERN = "optimal_configs_mi_top5_no_leak_report*.md"
OUTPUT_DIR = Path("visuals")
ALG_CHART = "mi_algorithm_metric_comparison.png"
BEST_CHART = "mi_best_config_comparison.png"


def safe_path(path):
    if not path.exists():
        return path
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def latest_report_path():
    reports = sorted(Path(".").glob(REPORT_PATTERN), key=lambda p: p.stat().st_mtime)
    if not reports:
        raise FileNotFoundError(f"Khong tim thay bao cao: {REPORT_PATTERN}")
    return reports[-1]


def parse_best_config_table(report_path):
    lines = report_path.read_text(encoding="utf-8").splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == "## Cau hinh tot nhat moi thuat toan":
            start = idx
            break
    if start is None:
        raise ValueError("Khong tim thay bang cau hinh tot nhat trong bao cao.")

    rows = []
    for line in lines[start + 1 :]:
        if not line.startswith("|"):
            continue
        if "---" in line or "Thuat toan" in line:
            continue

        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 7:
            continue
        algo, config, threshold, recall, precision, f1, auc = cells
        rows.append(
            {
                "Algorithm": algo,
                "Config": config,
                "Threshold": float(threshold),
                "Recall": float(recall.rstrip("%")),
                "Precision": float(precision.rstrip("%")),
                "F1": float(f1.rstrip("%")),
                "AUC": float(auc.rstrip("%")),
            }
        )

    if len(rows) != 5:
        raise ValueError(f"Bang best config khong du 5 dong, hien co {len(rows)} dong.")
    return pd.DataFrame(rows)


def add_bar_labels(ax, bars, suffix="%"):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}{suffix}",
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

    ax.set_title("So sanh cac thuat toan - MI Top 5 train-only", fontsize=15, weight="bold")
    ax.set_ylabel("Ty le (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Algorithm"], rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_best_config_comparison(df, output_path):
    plot_df = df.sort_values("F1", ascending=True).copy()
    labels = plot_df["Algorithm"] + "\n" + plot_df["Config"]
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(12, 7))
    f1_bars = ax.barh(y - 0.18, plot_df["F1"], height=0.34, label="F1", color="#f2994a")
    recall_bars = ax.barh(
        y + 0.18, plot_df["Recall"], height=0.34, label="Recall", color="#2f80ed"
    )

    for bars in (f1_bars, recall_bars):
        for bar in bars:
            width = bar.get_width()
            ax.annotate(
                f"{width:.2f}%",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
            )

    ax.set_title("So sanh cau hinh tot nhat moi thuat toan", fontsize=15, weight="bold")
    ax.set_xlabel("Ty le (%)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(55, 90)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    report_path = latest_report_path()
    df = parse_best_config_table(report_path)

    algorithm_chart = safe_path(OUTPUT_DIR / ALG_CHART)
    best_chart = safe_path(OUTPUT_DIR / BEST_CHART)

    plot_algorithm_metric_comparison(df, algorithm_chart)
    plot_best_config_comparison(df, best_chart)

    print(f"Da doc bao cao: {report_path}")
    print("Da tao bieu do:")
    print(f"- {algorithm_chart}")
    print(f"- {best_chart}")


if __name__ == "__main__":
    main()
