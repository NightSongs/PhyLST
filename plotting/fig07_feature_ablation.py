"""Fig 07 — Feature Ablation: R² and RMSE as separate bar charts."""

import numpy as np
import matplotlib.pyplot as plt
from plotting.plot_config import apply_global_style, savefig


def plot_fig07(abl_summary_df, feature_subsets, FIG_DIR):
    apply_global_style()

    subset_names = list(feature_subsets.keys())
    x_abl = np.arange(len(subset_names))
    abl_colors = ["#42A5F5", "#66BB6A", "#EF5350", "#FFA726", "#AB47BC", "#26A69A"]

    panels = [
        ("R2",   "R² by Feature Set",   "fig07a_ablation_r2.png"),
        ("RMSE", "RMSE by Feature Set",  "fig07b_ablation_rmse.png"),
    ]

    for metric, title, fname in panels:
        fig, ax = plt.subplots(figsize=(10, 6))
        means = [abl_summary_df.loc[abl_summary_df["subset"] == s, f"{metric}_mean"].values[0]
                 for s in subset_names]
        stds = [abl_summary_df.loc[abl_summary_df["subset"] == s, f"{metric}_std"].values[0]
                for s in subset_names]
        bars = ax.bar(x_abl, means, yerr=stds, color=abl_colors[:len(subset_names)],
                      alpha=0.9, edgecolor="white", capsize=3)
        ax.set_xticks(x_abl)
        ax.set_xticklabels(subset_names, rotation=35, ha="right")
        ax.set_title(title)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=14)
        savefig(fig, FIG_DIR / fname)

    print("  Fig 07: Feature ablation — done")
