"""Fig 04 — Best Model: density scatter, residual histogram, residual vs observed."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from plotting.plot_config import apply_global_style, add_equal_height_colorbar, savefig
from config import MODEL_SHORT


def plot_fig04(concat_preds, best_model_name, compute_metrics, FIG_DIR):
    apply_global_style()

    bp = concat_preds[best_model_name]
    y_true = bp["y_true"]
    y_pred = bp["y_pred"]
    m = compute_metrics(y_true, y_pred)
    short = MODEL_SHORT[best_model_name]

    # (a) Density scatter
    fig, ax = plt.subplots(figsize=(8, 7))
    h = ax.hist2d(y_true, y_pred, bins=200, cmap="magma_r", norm=LogNorm(), rasterized=True)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "w--", lw=1.5, alpha=0.8)
    ax.set_xlabel("Observed LST (°C)")
    ax.set_ylabel("Predicted LST (°C)")
    ax.set_title(f"{short}: R²={m['R2']:.4f}, RMSE={m['RMSE']:.2f}°C")
    add_equal_height_colorbar(fig, ax, h[3], label="Count")
    savefig(fig, FIG_DIR / "fig04a_density_scatter.png")

    # (b) Residual histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    residuals = y_pred - y_true
    ax.hist(residuals, bins=150, color="#FFA726", edgecolor="white", alpha=0.9, linewidth=0.3, density=True)
    ax.axvline(0, color="#EF5350", ls="--", lw=1.5)
    ax.axvline(m["Bias"], color="#42A5F5", ls="-", lw=1.5, label=f"Bias={m['Bias']:.3f}°C")
    ax.set_xlabel("Residual (°C)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")
    ax.legend()
    savefig(fig, FIG_DIR / "fig04b_residual_histogram.png")

    # (c) Residual vs observed
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, residuals, s=0.3, alpha=0.05, c="#607D8B", rasterized=True)
    ax.axhline(0, color="#EF5350", ls="--", lw=1.5)
    bins_obs = np.linspace(y_true.min(), y_true.max(), 50)
    bin_centers = 0.5 * (bins_obs[:-1] + bins_obs[1:])
    bin_means = []
    for j in range(len(bins_obs) - 1):
        mask = (y_true >= bins_obs[j]) & (y_true < bins_obs[j + 1])
        bin_means.append(residuals[mask].mean() if mask.sum() > 0 else np.nan)
    ax.plot(bin_centers, bin_means, "o-", color="#EF5350", ms=3, lw=1.5, label="Binned mean")
    ax.set_xlabel("Observed LST (°C)")
    ax.set_ylabel("Residual (°C)")
    ax.set_title("Residual vs Observed")
    ax.legend()
    savefig(fig, FIG_DIR / "fig04c_residual_vs_observed.png")

    print("  Fig 04: Best model scatter — done")
