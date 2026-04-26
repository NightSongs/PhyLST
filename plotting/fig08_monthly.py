"""Fig 08 — Monthly Performance: RMSE, R², MAE, Bias (separate panels)."""

import numpy as np
import matplotlib.pyplot as plt
from plotting.plot_config import apply_global_style, savefig
from config import MODEL_NAMES, MODEL_SHORT, MODEL_COLORS


def plot_fig08(concat_preds, compute_metrics, FIG_DIR):
    apply_global_style()

    panels = [
        ("RMSE", "RMSE (°C) by Month",  "fig08a_monthly_rmse.png"),
        ("R2",   "R² by Month",          "fig08b_monthly_r2.png"),
        ("MAE",  "MAE (°C) by Month",    "fig08c_monthly_mae.png"),
        ("Bias", "Bias (°C) by Month",   "fig08d_monthly_bias.png"),
    ]

    for metric, title, fname in panels:
        fig, ax = plt.subplots(figsize=(10, 6))
        for mn in MODEL_NAMES:
            mp = concat_preds[mn]
            monthly_vals = []
            for mo in range(1, 13):
                mask = mp["month"] == mo
                if mask.sum() > 0:
                    m = compute_metrics(mp["y_true"][mask], mp["y_pred"][mask])
                    monthly_vals.append(m[metric])
                else:
                    monthly_vals.append(np.nan)
            ax.plot(range(1, 13), monthly_vals, "o-", color=MODEL_COLORS[mn],
                    label=MODEL_SHORT[mn], lw=1.5, ms=4, alpha=0.85)
        ax.set_xlabel("Month")
        ax.set_ylabel(title.split(" by")[0])
        ax.set_title(title)
        ax.set_xticks(range(1, 13))
        ax.legend(ncol=2)
        if metric == "Bias":
            ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
        savefig(fig, FIG_DIR / fname)

    print("  Fig 08: Monthly performance — done")
