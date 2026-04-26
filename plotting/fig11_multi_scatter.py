"""Fig 11 — Multi-Model Density Scatter (one figure per model)."""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from plotting.plot_config import apply_global_style, add_equal_height_colorbar, savefig
from config import MODEL_SHORT


def plot_fig11(concat_preds, compute_metrics, FIG_DIR):
    apply_global_style()

    scatter_models = ["Ridge", "RandomForest", "XGBoost", "LightGBM", "ResNet1D", "PhyLST"]

    for i, mn in enumerate(scatter_models):
        fig, ax = plt.subplots(figsize=(8, 7))
        mp = concat_preds[mn]
        m = compute_metrics(mp["y_true"], mp["y_pred"])
        h = ax.hist2d(mp["y_true"], mp["y_pred"], bins=150, cmap="magma_r",
                      norm=LogNorm(), rasterized=True)
        vmin, vmax = mp["y_true"].min(), mp["y_true"].max()
        ax.plot([vmin, vmax], [vmin, vmax], "w--", lw=1.2, alpha=0.8)
        ax.set_xlabel("Observed (°C)")
        ax.set_ylabel("Predicted (°C)")
        ax.set_title(f"{MODEL_SHORT[mn]}  R²={m['R2']:.4f}  RMSE={m['RMSE']:.2f}°C  MAE={m['MAE']:.2f}°C")
        add_equal_height_colorbar(fig, ax, h[3], label="Count")
        savefig(fig, FIG_DIR / f"fig11{chr(97+i)}_scatter_{mn.lower()}.png")

    print("  Fig 11: Multi-model scatter — done")
