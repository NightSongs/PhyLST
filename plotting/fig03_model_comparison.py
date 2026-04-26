"""Fig 03 — Model Comparison: R², RMSE, MAE, |Bias| as separate bar charts."""

import numpy as np
import matplotlib.pyplot as plt
from plotting.plot_config import apply_global_style, savefig
from config import MODEL_NAMES, MODEL_SHORT, MODEL_COLORS


def plot_fig03(summary_df, FIG_DIR):
    apply_global_style()

    x = np.arange(len(MODEL_NAMES))
    bar_colors = [MODEL_COLORS[mn] for mn in MODEL_NAMES]

    panels = [
        ("R2",   "R²",           ".4f", False, "fig03a_r2.png"),
        ("RMSE", "RMSE (°C)",    ".3f", True,  "fig03b_rmse.png"),
        ("MAE",  "MAE (°C)",     ".3f", True,  "fig03c_mae.png"),
        ("Bias", "|Bias| (°C)",  ".4f", True,  "fig03d_bias.png"),
    ]

    for metric, title, fmt, invert, fname in panels:
        fig, ax = plt.subplots(figsize=(10, 6))
        if metric == "Bias":
            means = [abs(summary_df.loc[summary_df["model"] == mn, "Bias_mean"].values[0])
                     for mn in MODEL_NAMES]
            stds = [summary_df.loc[summary_df["model"] == mn, "Bias_std"].values[0]
                    for mn in MODEL_NAMES]
        else:
            means = [summary_df.loc[summary_df["model"] == mn, f"{metric}_mean"].values[0]
                     for mn in MODEL_NAMES]
            stds = [summary_df.loc[summary_df["model"] == mn, f"{metric}_std"].values[0]
                    for mn in MODEL_NAMES]

        bars = ax.bar(x, means, yerr=stds, color=bar_colors, alpha=0.9,
                      edgecolor="white", capsize=3, error_kw={"lw": 1, "capthick": 1})
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT[mn] for mn in MODEL_NAMES], rotation=30, ha="right")
        ax.set_title(title)

        best_i = np.argmin(means) if invert else np.argmax(means)
        bars[best_i].set_edgecolor("black")
        bars[best_i].set_linewidth(2)

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:{fmt}}", ha="center", va="bottom", fontsize=14)

        savefig(fig, FIG_DIR / fname)

    print("  Fig 03: Model comparison — done")
