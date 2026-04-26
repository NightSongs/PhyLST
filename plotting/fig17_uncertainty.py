"""Fig 17 — Uncertainty Analysis (PhyLST): scatter, spatial, seasonal."""

import numpy as np
import matplotlib.pyplot as plt
from plotting.plot_config import apply_global_style, add_equal_height_colorbar, savefig
from config import SEASON_ORDER, SEASON_LABELS, SEASON_COLORS


def plot_fig17(concat_preds, FIG_DIR):
    apply_global_style()

    pr = concat_preds["PhyLST"]
    pr_unc = pr["uncertainty"]
    pr_abs_err = np.abs(pr["y_pred"] - pr["y_true"])

    # (a) Uncertainty vs absolute error
    fig, ax = plt.subplots(figsize=(8, 7))
    h = ax.hexbin(pr_unc, pr_abs_err, gridsize=80, cmap="magma_r", mincnt=1, rasterized=True)
    add_equal_height_colorbar(fig, ax, h, label="Count")
    ax.set_xlabel("Predicted Uncertainty (%)")
    ax.set_ylabel("Absolute Error (°C)")
    ax.set_title("Uncertainty vs Error")
    valid = np.isfinite(pr_unc) & np.isfinite(pr_abs_err)
    if valid.sum() > 10:
        corr_val = np.corrcoef(pr_unc[valid], pr_abs_err[valid])[0, 1]
        ax.text(0.05, 0.95, f"r = {corr_val:.3f}", transform=ax.transAxes,
                va="top", fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    savefig(fig, FIG_DIR / "fig17a_uncertainty_vs_error.png")

    # (b) Uncertainty spatial map
    fig, ax = plt.subplots(figsize=(9, 8))
    step = max(1, len(pr_unc) // 50000)
    sc = ax.scatter(pr["lon"][::step], pr["lat"][::step],
                    c=pr_unc[::step], s=0.3, alpha=0.15,
                    cmap="YlOrRd", vmin=0, vmax=np.percentile(pr_unc, 95), rasterized=True)
    add_equal_height_colorbar(fig, ax, sc, label="Uncertainty (%)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Uncertainty Spatial Distribution")
    ax.set_aspect("equal")
    savefig(fig, FIG_DIR / "fig17b_uncertainty_spatial.png")

    # (c) Uncertainty by season
    fig, ax = plt.subplots(figsize=(8, 6))
    season_data = [pr_unc[pr["season"] == s] for s in SEASON_ORDER]
    bp = ax.boxplot(season_data, labels=[SEASON_LABELS[s] for s in SEASON_ORDER],
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="#EF5350", lw=1.5),
                    whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    for patch, season in zip(bp["boxes"], SEASON_ORDER):
        patch.set_facecolor(SEASON_COLORS[season])
        patch.set_edgecolor("gray")
        patch.set_linewidth(0.5)
    ax.set_ylabel("Uncertainty (%)")
    ax.set_title("Uncertainty by Season")
    savefig(fig, FIG_DIR / "fig17c_uncertainty_by_season.png")

    print("  Fig 17: Uncertainty analysis — done")
