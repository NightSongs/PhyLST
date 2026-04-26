"""Fig 12 — Seasonal Scatter for Best Model (one figure per season)."""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from plotting.plot_config import apply_global_style, add_equal_height_colorbar, savefig
from config import SEASON_ORDER, SEASON_LABELS, MODEL_SHORT


def plot_fig12(concat_preds, best_model_name, compute_metrics, FIG_DIR):
    apply_global_style()

    bp = concat_preds[best_model_name]
    short = MODEL_SHORT[best_model_name]

    for si, season in enumerate(SEASON_ORDER):
        fig, ax = plt.subplots(figsize=(8, 7))
        mask = bp["season"] == season
        yt = bp["y_true"][mask]
        yp = bp["y_pred"][mask]
        m = compute_metrics(yt, yp)

        h = ax.hist2d(yt, yp, bins=100, cmap="magma_r", norm=LogNorm(), rasterized=True)
        vmin, vmax = yt.min(), yt.max()
        ax.plot([vmin, vmax], [vmin, vmax], "w--", lw=1.2)
        ax.set_xlabel("Observed (°C)")
        ax.set_ylabel("Predicted (°C)")
        ax.set_title(f"{short} — {SEASON_LABELS[season]} ({season})\n"
                     f"R²={m['R2']:.4f}  RMSE={m['RMSE']:.2f}°C  Bias={m['Bias']:+.2f}°C  n={mask.sum():,}")
        add_equal_height_colorbar(fig, ax, h[3], label="Count")
        savefig(fig, FIG_DIR / f"fig12{chr(97+si)}_seasonal_{season.lower()}.png")

    print("  Fig 12: Seasonal scatter — done")
