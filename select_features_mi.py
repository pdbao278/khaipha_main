import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


INPUT_FILE = "data_processed.csv"
TARGET_COL = "Stroke"
OUTPUT_FILE = "data_mi_selected.csv"
SCORES_FILE = "mi_feature_scores.csv"
PLOT_FILE = os.path.join("visuals", "5_mutual_information.png")
MI_THRESHOLD = 0.01
CONTINUOUS_BINS = 10


def discretize_feature(series):
    if series.nunique(dropna=True) <= 10:
        return series.fillna("missing").astype(str)

    ranked = series.rank(method="first")
    return pd.qcut(
        ranked,
        q=min(CONTINUOUS_BINS, series.nunique(dropna=True)),
        duplicates="drop",
    ).astype(str)


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
                score += pxy * np.log(pxy / (px.loc[feature_value] * py.loc[target_value]))
    return float(score)


def load_font(size):
    font_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in font_candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_mi_chart(score_df):
    width = 1500
    height = max(780, 100 + len(score_df) * 62)
    left = 330
    right = 80
    top = 90
    row_h = 58
    bar_h = 34
    axis_w = width - left - right

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(30)
    label_font = load_font(22)
    small_font = load_font(18)

    title = "Feature Selection: Mutual Information vs Stroke Target"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_box[2] - title_box[0])) / 2, 24), title, fill="#222222", font=title_font)

    max_score = max(float(score_df["MI_Score"].max()), MI_THRESHOLD) * 1.08
    tick_count = 8
    for i in range(tick_count + 1):
        x = left + axis_w * i / tick_count
        draw.line((x, top, x, height - 90), fill="#d0d0d0", width=2)
        tick_value = max_score * i / tick_count
        draw.text((x - 22, height - 70), f"{tick_value:.2f}", fill="#333333", font=small_font)

    threshold_x = left + axis_w * MI_THRESHOLD / max_score
    draw.line((threshold_x, top, threshold_x, height - 90), fill="red", width=2)

    for idx, row in score_df.iterrows():
        y = top + idx * row_h
        feature = row["Feature"]
        score = float(row["MI_Score"])
        bar_w = axis_w * score / max_score
        color = "#2ecc71" if score >= MI_THRESHOLD else "#e74c3c"

        draw.text((left - 20, y + 5), feature, fill="#222222", font=label_font, anchor="ra")
        draw.rectangle((left, y, left + bar_w, y + bar_h), fill=color)
        draw.line((left, y + row_h - 10, width - right, y + row_h - 10), fill="#d0d0d0", width=2)

    x_label = "Mutual Information Score"
    x_label_box = draw.textbbox((0, 0), x_label, font=label_font)
    draw.text(((width - (x_label_box[2] - x_label_box[0])) / 2, height - 38), x_label, fill="#222222", font=label_font)

    legend_text = f"Threshold = {MI_THRESHOLD}"
    legend_w = 250
    legend_h = 44
    legend_x = width - right - legend_w
    legend_y = height - 112
    draw.rectangle((legend_x, legend_y, legend_x + legend_w, legend_y + legend_h), outline="#cccccc", width=2)
    draw.line((legend_x + 18, legend_y + 22, legend_x + 70, legend_y + 22), fill="red", width=2)
    draw.text((legend_x + 84, legend_y + 10), legend_text, fill="#333333", font=small_font)

    image.save(PLOT_FILE)


def main():
    os.makedirs("visuals", exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Khong tim thay cot muc tieu: {TARGET_COL}")

    x = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    mi_scores = [
        mutual_information_score(discretize_feature(x[col]), y)
        for col in x.columns
    ]

    score_df = (
        pd.DataFrame({"Feature": x.columns, "MI_Score": mi_scores})
        .sort_values("MI_Score", ascending=False)
        .reset_index(drop=True)
    )
    score_df.to_csv(SCORES_FILE, index=False)

    selected_features = score_df.loc[
        score_df["MI_Score"] >= MI_THRESHOLD, "Feature"
    ].tolist()
    if not selected_features:
        raise ValueError("Khong co dac trung nao vuot nguong MI da chon.")

    df_selected = df[selected_features + [TARGET_COL]]
    df_selected.to_csv(OUTPUT_FILE, index=False)

    draw_mi_chart(score_df)

    print("Da tao bo du lieu MI:")
    print(f"- File du lieu: {OUTPUT_FILE}")
    print(f"- File diem MI: {SCORES_FILE}")
    print(f"- Bieu do: {PLOT_FILE}")
    print(f"- So dac trung duoc chon: {len(selected_features)}")
    print("- Danh sach dac trung:")
    for feature in selected_features:
        print(f"  + {feature}")


if __name__ == "__main__":
    main()
