"""Fig 13 — Error CDF (all models)."""

import numpy as np
import matplotlib.pyplot as plt
from plotting.plot_config import apply_global_style, savefig
from config import MODEL_NAMES, MODEL_SHORT, MODEL_COLORS


def plot_fig13(concat_preds, FIG_DIR):
    apply_global_style()

    fig, ax = plt.subplots(figsize=(10, 7))
    for mn in MODEL_NAMES:
        mp = concat_preds[mn]
        abs_err = np.abs(mp["y_pred"] - mp["y_true"])
        sorted_err = np.sort(abs_err)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        step = max(1, len(sorted_err) // 5000)
        ax.plot(sorted_err[::step], cdf[::step], color=MODEL_COLORS[mn],
                label=MODEL_SHORT[mn], lw=1.5)

    ax.set_xlabel("Absolute Error (°C)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Cumulative Distribution of Absolute Errors")
    ax.axhline(0.9, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(0.95, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.text(ax.get_xlim()[1] * 0.95, 0.9, "90%", ha="right", va="bottom", color="gray")
    ax.text(ax.get_xlim()[1] * 0.95, 0.95, "95%", ha="right", va="bottom", color="gray")
    ax.legend()
    ax.set_xlim(0, 10)
    savefig(fig, FIG_DIR / "fig13_error_cdf.png")

    print("  Fig 13: Error CDF — done")
