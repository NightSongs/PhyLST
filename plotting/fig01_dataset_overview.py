"""Fig 01 — Dataset Overview: LST distribution, monthly boxplot, yearly bar."""

import numpy as np
import matplotlib.pyplot as plt
from plotting.plot_config import apply_global_style, savefig


def plot_fig01(df, TARGET, FIG_DIR):
    apply_global_style()

    # (a) LST Distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df[TARGET], bins=100, color="#42A5F5", edgecolor="white", alpha=0.9, linewidth=0.3)
    ax.set_xlabel("LST (°C)")
    ax.set_ylabel("Count")
    ax.set_title("LST Distribution")
    ax.axvline(df[TARGET].mean(), color="#EF5350", ls="--", lw=1.5,
               label=f"Mean = {df[TARGET].mean():.1f}°C")
    ax.axvline(df[TARGET].median(), color="#FFA726", ls=":", lw=1.5,
               label=f"Median = {df[TARGET].median():.1f}°C")
    ax.legend()
    savefig(fig, FIG_DIR / "fig01a_lst_distribution.png")

    # (b) LST by Month
    fig, ax = plt.subplots(figsize=(8, 6))
    month_data = [df[df["month"] == m][TARGET].values for m in range(1, 13)]
    bp = ax.boxplot(month_data, labels=range(1, 13), patch_artist=True,
                    showfliers=False, medianprops=dict(color="#EF5350", lw=1.5),
                    whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    colors_month = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, 12))
    for patch, color in zip(bp["boxes"], colors_month):
        patch.set_facecolor(color)
        patch.set_edgecolor("gray")
        patch.set_linewidth(0.5)
    ax.set_xlabel("Month")
    ax.set_ylabel("LST (°C)")
    ax.set_title("LST by Month")
    savefig(fig, FIG_DIR / "fig01b_lst_by_month.png")

    # (c) Temporal Distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    year_counts = df["year"].value_counts().sort_index()
    ax.bar(year_counts.index, year_counts.values, color="#66BB6A", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Sample Count")
    ax.set_title("Temporal Distribution")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(4, 4))
    savefig(fig, FIG_DIR / "fig01c_temporal_distribution.png")

    print("  Fig 01: Dataset overview — done")
