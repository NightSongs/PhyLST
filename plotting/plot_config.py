"""
=============================================================================
Global Plotting Configuration
=============================================================================
All figures use:
  - Font: Times New Roman
  - Font size: 18
  - No multi-panel subplots — each panel is a separate canvas/file
  - Scatter colorbars are always the same height as the plot area
  - 300 DPI for saved figures
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def apply_global_style():
    """Apply global matplotlib style for all figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 14,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.fontset": "stix",
    })


def add_equal_height_colorbar(fig, ax, mappable, label=""):
    """Add a colorbar that is exactly the same height as the axes.

    Uses make_axes_locatable to create a colorbar axis that matches
    the height of the plot area.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure.
    ax : Axes
        The axes containing the mappable.
    mappable : ScalarMappable
        The image/scatter/hexbin to attach the colorbar to.
    label : str
        Colorbar label.

    Returns
    -------
    cbar : Colorbar
        The created colorbar.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax, label=label)
    return cbar


def savefig(fig, path):
    """Save figure with tight bounding box and close."""
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"    Saved: {path.name}")
