"""Fig 14 — Taylor Diagram."""

import numpy as np
import matplotlib.pyplot as plt
from plotting.plot_config import apply_global_style, savefig
from config import MODEL_NAMES, MODEL_SHORT, MODEL_COLORS


def plot_fig14(concat_preds, FIG_DIR):
    apply_global_style()

    ref_std = np.std(concat_preds[MODEL_NAMES[0]]["y_true"])
    max_std = ref_std * 1.6

    fig, ax = plt.subplots(figsize=(9, 9))

    # Reference point
    ax.plot(ref_std, 0, "ko", ms=10, label="Observed", zorder=5)

    # Reference arc
    theta = np.linspace(0, np.pi / 2, 100)
    ax.plot(ref_std * np.cos(theta), ref_std * np.sin(theta), "k--", lw=0.8, alpha=0.3)

    # Correlation lines
    for corr_val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        angle = np.arccos(corr_val)
        ax.plot([0, max_std * np.cos(angle)], [0, max_std * np.sin(angle)],
                "gray", lw=0.3, alpha=0.3)
        ax.text(max_std * np.cos(angle) * 1.02, max_std * np.sin(angle) * 1.02,
                f"{corr_val}", fontsize=10, color="gray", ha="center")

    # RMSE arcs
    for rmse_val in np.arange(0.5, max_std, 0.5):
        circle_theta = np.linspace(0, np.pi, 200)
        cx = ref_std + rmse_val * np.cos(circle_theta)
        cy = rmse_val * np.sin(circle_theta)
        valid = (cx >= 0) & (cy >= 0)
        ax.plot(cx[valid], cy[valid], ":", color="green", lw=0.4, alpha=0.3)

    # Model points
    for mn in MODEL_NAMES:
        mp = concat_preds[mn]
        pred_std = np.std(mp["y_pred"])
        corr = np.corrcoef(mp["y_true"], mp["y_pred"])[0, 1]
        angle = np.arccos(np.clip(corr, -1, 1))
        x = pred_std * np.cos(angle)
        y = pred_std * np.sin(angle)
        ax.plot(x, y, "o", color=MODEL_COLORS[mn], ms=10, label=MODEL_SHORT[mn],
                zorder=4, markeredgecolor="white", markeredgewidth=0.5)

    ax.set_xlim(0, max_std)
    ax.set_ylim(0, max_std)
    ax.set_xlabel("Standard Deviation (°C)")
    ax.set_ylabel("Standard Deviation (°C)")
    ax.set_title("Taylor Diagram — All Models")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    savefig(fig, FIG_DIR / "fig14_taylor_diagram.png")

    print("  Fig 14: Taylor diagram — done")
