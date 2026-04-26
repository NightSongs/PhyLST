"""
=============================================================================
DL Baseline Training — standard training for MLP and ResNet1D
=============================================================================
No Mixup, no SWA, no sample weighting. Conservative baseline configuration
commonly seen in remote sensing literature.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from . import EarlyStopping, lr_cosine_warmup


def train_dl_model_baseline(model, X_train, y_train, X_val, y_val, y_mean, y_std,
                            epochs=150, batch_size=2048, lr=1e-3, weight_decay=1e-4,
                            patience=15, verbose_every=50, warmup_epochs=5,
                            device=None):
    """Train a baseline DL model (MLP or ResNet1D) with standard settings.

    Returns
    -------
    model : nn.Module
        Trained model (loaded with best state).
    history : dict
        Training history (train_loss, val_loss, val_rmse).
    best_state : dict
        Best model state_dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_n, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val_n, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: lr_cosine_warmup(e, warmup_epochs, epochs)
    )

    criterion = nn.HuberLoss(delta=1.0)
    es = EarlyStopping(patience=patience)

    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_rmse": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        model.eval()
        val_losses, val_preds = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())
                val_preds.append(pred.cpu().numpy())

        val_preds_all = np.concatenate(val_preds) * y_std + y_mean
        val_rmse = np.sqrt(np.mean((y_val - val_preds_all) ** 2))

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_rmse"].append(val_rmse)

        improved = es.step(avg_val)
        if improved:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % verbose_every == 0 or epoch == 1:
            print(f"      Epoch {epoch:3d}: train={avg_train:.5f}, val={avg_val:.5f}, RMSE={val_rmse:.3f}")

        if es.should_stop:
            print(f"      Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    return model, history, best_state


def predict_dl(model, X, y_mean, y_std, batch_size=4096, device=None):
    """Generate predictions from a baseline DL model.

    Returns ndarray of predictions in original (Celsius) space.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds) * y_std + y_mean
