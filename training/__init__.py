"""Training utilities: EarlyStopping, loss functions, LR schedule helpers."""

import numpy as np
import torch
import torch.nn as nn


class EarlyStopping:
    """Stops training when validation loss stops improving."""

    def __init__(self, patience=10, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def step(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


class WeightedHuberLoss(nn.Module):
    """Huber loss with per-sample weights."""

    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target, weights=None):
        diff = pred - target
        abs_diff = torch.abs(diff)
        quadratic = torch.clamp(abs_diff, max=self.delta)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        if weights is not None:
            loss = loss * weights
        return loss.mean()


def nll_loss_fn(pred, log_var, target, weights=None):
    """Negative log-likelihood for heteroscedastic Gaussian regression."""
    precision = torch.exp(-log_var)
    loss = 0.5 * (log_var + precision * (target - pred) ** 2)
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def lr_cosine_warmup(epoch, warmup_epochs, epochs):
    """Cosine annealing with linear warmup. Returns multiplier in [0, 1]."""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))
