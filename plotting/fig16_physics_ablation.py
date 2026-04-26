"""Fig 16 — Physics Constraint Ablation: R², RMSE, MAE (separate panels)."""

import numpy as np
import matplotlib.pyplot as plt
from plotting.plot_config import apply_global_style, savefig
from config import ABLATION_CONFIGS


def plot_fig16(phys_abl_summary_df, FIG_DIR):
    apply_global_style()

    abl_names = list(ABLATION_CONFIGS.keys())
    abl_short = ["NoPhys", "MonoOnly", "Full"]
    abl_colors = ["#90A4AE", "#FFA726", "#EF5350"]
    x_abl = np.arange(len(abl_names))

    panels = [
        ("R2",   "R²",         "fig16a_phys_ablation_r2.png"),
        ("RMSE", "RMSE (°C)",  "fig16b_phys_ablation_rmse.png"),
        ("MAE",  "MAE (°C)",   "fig16c_phys_ablation_mae.png"),
    ]

    for metric, title, fname in panels:
        fig, ax = plt.subplots(figsize=(8, 6))
        means = [phys_abl_summary_df.loc[phys_abl_summary_df["ablation"] == n, f"{metric}_mean"].values[0]
                 for n in abl_names]
        stds = [phys_abl_summary_df.loc[phys_abl_summary_df["ablation"] == n, f"{metric}_std"].values[0]
                for n in abl_names]
        bars = ax.bar(x_abl, means, yerr=stds, color=abl_colors, alpha=0.9,
                      edgecolor="white", capsize=4, error_kw={"lw": 1, "capthick": 1})
        ax.set_xticks(x_abl)
        ax.set_xticklabels(abl_short)
        ax.set_title(f"Physics Constraint Ablation — {title}")
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=14)
        savefig(fig, FIG_DIR / fname)

    print("  Fig 16: Physics ablation — done")
