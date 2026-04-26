"""Fig 15 — Residual Spatial Map (best model): residual + absolute error."""

import matplotlib.pyplot as plt
from plotting.plot_config import apply_global_style, add_equal_height_colorbar, savefig
from config import MODEL_SHORT


def plot_fig15(concat_preds, best_model_name, FIG_DIR):
    apply_global_style()

    bp = concat_preds[best_model_name]
    residuals = bp["y_pred"] - bp["y_true"]
    import numpy as np
    abs_err = np.abs(residuals)
    short = MODEL_SHORT[best_model_name]

    # (a) Residual spatial
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(bp["lon"], bp["lat"], c=residuals, s=0.2, alpha=0.1,
                    cmap="RdBu_r", vmin=-5, vmax=5, rasterized=True)
    add_equal_height_colorbar(fig, ax, sc, label="Residual (°C)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"{short} Residual Spatial Distribution")
    ax.set_aspect("equal")
    savefig(fig, FIG_DIR / "fig15a_residual_spatial.png")

    # (b) Absolute error spatial
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(bp["lon"], bp["lat"], c=abs_err, s=0.2, alpha=0.1,
                    cmap="YlOrRd", vmin=0, vmax=5, rasterized=True)
    add_equal_height_colorbar(fig, ax, sc, label="|Error| (°C)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"{short} Absolute Error Spatial Distribution")
    ax.set_aspect("equal")
    savefig(fig, FIG_DIR / "fig15b_abserror_spatial.png")

    print("  Fig 15: Residual spatial map — done")
