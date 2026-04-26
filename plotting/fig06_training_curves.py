"""Fig 06 — Training Curves: train loss and validation RMSE (separate)."""

import numpy as np
import matplotlib.pyplot as plt
from plotting.plot_config import apply_global_style, savefig
from config import MODEL_SHORT


def plot_fig06(dl_histories, FIG_DIR):
    apply_global_style()

    dl_plot_colors = {"MLP": "#42A5F5", "ResNet1D": "#26A69A", "PhyLST": "#EF5350"}

    # (a) Training Loss
    fig, ax = plt.subplots(figsize=(9, 6))
    for dl_name, color in dl_plot_colors.items():
        histories = dl_histories[dl_name]
        max_len = max(len(h["train_loss"]) for h in histories)
        arrs = np.array([np.pad(h["train_loss"], (0, max_len - len(h["train_loss"])),
                                constant_values=np.nan) for h in histories])
        mean = np.nanmean(arrs, axis=0)
        std = np.nanstd(arrs, axis=0)
        ep = np.arange(1, max_len + 1)
        ax.plot(ep, mean, color=color, label=MODEL_SHORT[dl_name], lw=1.5)
        ax.fill_between(ep, mean - std, mean + std, color=color, alpha=0.12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.set_ylim(bottom=0)
    savefig(fig, FIG_DIR / "fig06a_training_loss.png")

    # (b) Validation RMSE
    fig, ax = plt.subplots(figsize=(9, 6))
    for dl_name, color in dl_plot_colors.items():
        histories = dl_histories[dl_name]
        max_len = max(len(h["val_rmse"]) for h in histories)
        arrs = np.array([np.pad(h["val_rmse"], (0, max_len - len(h["val_rmse"])),
                                constant_values=np.nan) for h in histories])
        mean = np.nanmean(arrs, axis=0)
        std = np.nanstd(arrs, axis=0)
        ep = np.arange(1, max_len + 1)
        ax.plot(ep, mean, color=color, label=MODEL_SHORT[dl_name], lw=1.5)
        ax.fill_between(ep, mean - std, mean + std, color=color, alpha=0.12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE (°C)")
    ax.set_title("Validation RMSE")
    ax.legend()
    savefig(fig, FIG_DIR / "fig06b_validation_rmse.png")

    print("  Fig 06: Training curves — done")
