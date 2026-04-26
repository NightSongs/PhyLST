"""Fig 02 — Correlation Heatmap (non-spectral features)."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotting.plot_config import apply_global_style, savefig


def plot_fig02(df, TOPO_COLS, NDVI_COLS, METEO_COLS, TEMPORAL_COLS, COORD_COLS, TARGET, FIG_DIR):
    apply_global_style()

    non_a_feats = TOPO_COLS + NDVI_COLS + METEO_COLS + TEMPORAL_COLS + COORD_COLS + [TARGET]
    corr_mat = df[non_a_feats].corr()

    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, ax=ax,
                annot_kws={"size": 10}, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix (non-spectral features)")
    savefig(fig, FIG_DIR / "fig02_correlation_heatmap.png")

    print("  Fig 02: Correlation heatmap — done")
