"""Fig 05 — Random Forest Feature Importance (top 30)."""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from plotting.plot_config import apply_global_style, savefig
from config import METEO_COLS, TOPO_COLS, NDVI_COLS, TEMPORAL_COLS


def _get_feature_color(f):
    if f in METEO_COLS: return "#EF5350"
    if f.startswith("A"): return "#42A5F5"
    if f in TOPO_COLS: return "#66BB6A"
    if f in NDVI_COLS: return "#FFA726"
    if f in TEMPORAL_COLS: return "#AB47BC"
    if f.startswith("IX_"): return "#26A69A"
    return "#795548"


def plot_fig05(rf_importance_df, FIG_DIR, top_n=30):
    apply_global_style()

    fig, ax = plt.subplots(figsize=(10, 10))
    rf_top = rf_importance_df.head(top_n)
    colors_fi = [_get_feature_color(f) for f in rf_top["feature"]]

    ax.barh(range(top_n), rf_top["importance_mean"].values[::-1],
            xerr=rf_top["importance_std"].values[::-1],
            color=colors_fi[::-1], capsize=3, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(rf_top["feature"].values[::-1], fontsize=14)
    ax.set_xlabel("Feature Importance (mean ± std)")
    ax.set_title(f"Random Forest Feature Importance (Top {top_n})")
    legend_elements = [
        Patch(facecolor="#EF5350", label="Meteorological"),
        Patch(facecolor="#42A5F5", label="Spectral (A00-A63)"),
        Patch(facecolor="#66BB6A", label="Topographic"),
        Patch(facecolor="#FFA726", label="NDVI"),
        Patch(facecolor="#AB47BC", label="Temporal"),
        Patch(facecolor="#26A69A", label="Seasonal Interaction"),
        Patch(facecolor="#795548", label="Coordinate"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)
    savefig(fig, FIG_DIR / "fig05_feature_importance.png")

    print("  Fig 05: Feature importance — done")
