"""Fig 10 — Spatial Distribution of Samples."""

import matplotlib.pyplot as plt
from plotting.plot_config import apply_global_style, add_equal_height_colorbar, savefig


def plot_fig10(df, TARGET, FIG_DIR):
    apply_global_style()

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(df["lon"], df["lat"], c=df[TARGET], s=0.1, alpha=0.05,
                    cmap="RdYlBu_r", vmin=10, vmax=45, rasterized=True)
    add_equal_height_colorbar(fig, ax, sc, label="LST (°C)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Spatial Distribution of Samples")
    ax.set_aspect("equal")
    savefig(fig, FIG_DIR / "fig10_spatial_distribution.png")

    print("  Fig 10: Spatial distribution — done")
