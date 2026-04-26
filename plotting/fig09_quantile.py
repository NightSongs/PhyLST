"""Fig 09 — Quantile Performance: RMSE, MAE, Bias (separate panels)."""

import numpy as np
import matplotlib.pyplot as plt
from plotting.plot_config import apply_global_style, savefig
from config import MODEL_NAMES, MODEL_SHORT, MODEL_COLORS


def plot_fig09(concat_preds, compute_metrics, FIG_DIR):
    apply_global_style()

    quantile_labels = ["Q1\n(cold)", "Q2", "Q3", "Q4", "Q5\n(hot)"]
    ref_y = concat_preds[MODEL_NAMES[0]]["y_true"]
    quantile_edges = np.quantile(ref_y, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    x_q = np.arange(5)
    bar_width = 0.12

    panels = [
        ("RMSE", "RMSE (°C) by Quantile", "fig09a_quantile_rmse.png"),
        ("MAE",  "MAE (°C) by Quantile",  "fig09b_quantile_mae.png"),
        ("Bias", "Bias (°C) by Quantile",  "fig09c_quantile_bias.png"),
    ]

    for metric, title, fname in panels:
        fig, ax = plt.subplots(figsize=(12, 6))
        for j, mn in enumerate(MODEL_NAMES):
            mp = concat_preds[mn]
            vals = []
            for qi in range(5):
                lo, hi = quantile_edges[qi], quantile_edges[qi + 1]
                mask = (mp["y_true"] >= lo) & (mp["y_true"] < hi) if qi < 4 else (mp["y_true"] >= lo)
                if mask.sum() > 0:
                    m = compute_metrics(mp["y_true"][mask], mp["y_pred"][mask])
                    vals.append(m[metric])
                else:
                    vals.append(np.nan)
            offset = (j - len(MODEL_NAMES) / 2 + 0.5) * bar_width
            ax.bar(x_q + offset, vals, bar_width, color=MODEL_COLORS[mn],
                   label=MODEL_SHORT[mn], alpha=0.9, edgecolor="white", linewidth=0.3)
        ax.set_xticks(x_q)
        ax.set_xticklabels(quantile_labels)
        ax.set_title(title)
        ax.legend(ncol=2)
        if metric == "Bias":
            ax.axhline(0, color="gray", ls="--", lw=0.8)
        savefig(fig, FIG_DIR / fname)

    print("  Fig 09: Quantile performance — done")
