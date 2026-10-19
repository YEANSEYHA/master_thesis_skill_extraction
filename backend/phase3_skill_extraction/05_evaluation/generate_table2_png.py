"""
Generate publication-quality PNG of Table 2 (Hard F1 across thresholds).

Usage:
    cd backend/phase3_skill_extraction/evaluation
    python generate_table2_png.py

Output:
    table2_hard_f1_thresholds.png (300 DPI, ~600KB)
"""

import matplotlib.pyplot as plt


# ── Table data ──
COLUMNS = ["Threshold", "KeyBERT", "qwen3:14b (5–10)", "qwen3:14b (5–15)"]

# Row format: (threshold, [keybert, qwen_5_10, qwen_5_15], best_index)
# best_index = which column (1-based, ignoring "Threshold") has the best value -> bold
ROWS = [
    ("@0.60", ["39.2%", "47.1%", "50.1%"], 3),  # qwen 5-15 wins
    ("@0.65", ["28.4%", "45.2%", "47.8%"], 3),
    ("@0.70", ["19.5%", "41.6%", "43.7%"], 3),
    ("@0.75", ["12.6%", "38.1%", "39.5%"], 3),
    ("@0.80", ["8.2%",  "33.7%", "34.8%"], 3),
]


def main():
    fig, ax = plt.subplots(figsize=(11, 3.5), dpi=300)
    ax.axis("off")

    cell_text = []
    cell_weights = []
    for thr, values, best_idx in ROWS:
        cell_text.append([thr] + values)
        weights = ["bold"] + ["normal"] * 3
        weights[best_idx] = "bold"
        cell_weights.append(weights)

    table = ax.table(
        cellText=cell_text,
        colLabels=COLUMNS,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.0, 2.2)

    n_rows = len(ROWS) + 1
    n_cols = len(COLUMNS)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)
        if row == 0:
            cell.set_text_props(weight="bold", fontsize=13)
            cell.set_facecolor("#f0f0f0")
        else:
            weight = cell_weights[row - 1][col]
            cell.set_text_props(weight=weight, fontsize=12)
            if col == 0:
                cell.set_text_props(ha="left", weight=weight, fontsize=12)
                cell.PAD = 0.05
            else:
                cell.set_text_props(ha="center", weight=weight, fontsize=12)

    for col in range(n_cols):
        table[(0, col)].visible_edges = "TB"
        table[(0, col)].set_linewidth(1.2)
        table[(n_rows - 1, col)].visible_edges = "B"
        table[(n_rows - 1, col)].set_linewidth(1.2)

    for col in range(n_cols):
        cell = table[(0, col)]
        cell.set_edgecolor("black")
        cell.set_linewidth(1.0)

    plt.tight_layout()

    out_path = "table2_hard_f1_thresholds.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {out_path}")
    print(f"   Resolution: 300 DPI, print-ready")
    print(f"   Open with: open {out_path}")


if __name__ == "__main__":
    main()
