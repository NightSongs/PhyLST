"""
=============================================================================
PhyLST Training — physics-constrained training with enhanced priors
=============================================================================
Training loop for PhyLST with:
  - NLL loss (heteroscedastic uncertainty)
  - Expanded monotonicity (positive + negative)
  - Radiation-driven diurnal constraint
  - Spatial regularization via feature similarity
  - Mixup data augmentation
  - SWA (Stochastic Weight Averaging)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel, SWALR

from . import EarlyStopping, nll_loss_fn, lr_cosine_warmup


def train_phylst(model, X_train, y_train, X_val, y_val, y_mean, y_std,
                 pos_mono_indices, neg_mono_indices,
                 t2m_feat_idx, ssrd_feat_idx, n_static,
                 t2m_feat_mean, t2m_feat_std,
                 mono_weight=0.003, mono_margin=0.01,
                 rad_weight=0.002, spatial_weight=0.001,
                 sample_weights=None,
                 epochs=300, batch_size=2048, lr=7e-4, weight_decay=1e-4,
                 patience=35, verbose_every=50, use_swa=True, swa_start_frac=0.75,
                 mixup_alpha=0.1, warmup_epochs=8, device=None):
    """Train PhyLST with enhanced physics constraints.

    Parameters
    ----------
    model : PhyLST
        The PhyLST model instance.
    X_train, y_train : ndarray
        Standardized training features and raw targets.
    X_val, y_val : ndarray
        Standardized validation features and raw targets.
    y_mean, y_std : float
        Target normalization parameters.
    pos_mono_indices : list[int]
        Feature indices with positive monotonicity constraint.
    neg_mono_indices : list[int]
        Feature indices with negative monotonicity constraint.
    t2m_feat_idx : int
        Index of t2m_mean in feature vector.
    ssrd_feat_idx : int
        Index of ssrd_daily in feature vector.
    n_static : int
        Number of static features.
    t2m_feat_mean, t2m_feat_std : float
        Scaler stats for t2m_mean feature.
    mono_weight : float
        Weight for monotonicity loss.
    mono_margin : float
        Margin for monotonicity penalty.
    rad_weight : float
        Weight for radiation constraint.
    spatial_weight : float
        Weight for spatial regularization.
    sample_weights : ndarray or None
        Per-sample weights.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Initial learning rate.
    weight_decay : float
        L2 regularization.
    patience : int
        Early stopping patience.
    verbose_every : int
        Print frequency.
    use_swa : bool
        Whether to use Stochastic Weight Averaging.
    swa_start_frac : float
        Fraction of epochs after which SWA starts.
    mixup_alpha : float
        Mixup interpolation parameter (0 = disabled).
    warmup_epochs : int
        Linear warmup epochs.
    device : torch.device or None
        Compute device.

    Returns
    -------
    model : PhyLST
        Trained model (loaded with best state).
    history : dict
        Training history.
    best_state : dict
        Best model state_dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    if sample_weights is not None:
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train_n, dtype=torch.float32),
            torch.tensor(sample_weights, dtype=torch.float32),
        )
    else:
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

    val_criterion = nn.HuberLoss(delta=1.0)

    swa_start = int(epochs * swa_start_frac)
    swa_model = AveragedModel(model) if use_swa else None
    swa_scheduler = SWALR(optimizer, swa_lr=lr * 0.1, anneal_epochs=5) if use_swa else None

    es = EarlyStopping(patience=patience)
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_rmse": [],
               "mono_loss": [], "rad_loss": [], "spatial_loss": []}

    pos_mono_indices_t = torch.tensor(pos_mono_indices, dtype=torch.long, device=device)
    neg_mono_indices_t = torch.tensor(neg_mono_indices, dtype=torch.long, device=device)
    t2m_feat_mean_t = torch.tensor(t2m_feat_mean, dtype=torch.float32, device=device)
    t2m_feat_std_t = torch.tensor(t2m_feat_std, dtype=torch.float32, device=device)
    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.tensor(y_std, dtype=torch.float32, device=device)

    has_weights = sample_weights is not None
    mono_freq = 5
    spatial_freq = 10

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        mono_losses = []
        rad_losses = []
        spatial_losses = []

        for batch_idx, batch in enumerate(train_loader):
            if has_weights:
                xb, yb, wb = batch
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            else:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                wb = None

            is_physics_batch = (batch_idx % mono_freq == 0 and epoch > warmup_epochs)
            is_spatial_batch = (batch_idx % spatial_freq == 0 and epoch > warmup_epochs
                                and spatial_weight > 0)

            if is_physics_batch and (mono_weight > 0 or rad_weight > 0):
                xb.requires_grad_(True)
                pred, log_var = model(xb)
                loss_main = nll_loss_fn(pred, log_var, yb, wb)

                total_physics = torch.tensor(0.0, device=device)

                # Expanded monotonicity
                if mono_weight > 0:
                    grad = torch.autograd.grad(pred.sum(), xb, create_graph=True)[0]
                    pos_grads = grad[:, pos_mono_indices_t]
                    mono_penalty = torch.relu(-pos_grads - mono_margin).mean()
                    if len(neg_mono_indices) > 0:
                        neg_grads = grad[:, neg_mono_indices_t]
                        mono_penalty = mono_penalty + torch.relu(neg_grads - mono_margin).mean()
                    mono_losses.append(mono_penalty.item())
                    total_physics = total_physics + mono_weight * mono_penalty

                # Radiation-driven constraint
                if rad_weight > 0:
                    t2m_orig = xb[:, t2m_feat_idx] * t2m_feat_std_t + t2m_feat_mean_t
                    lst_pred_c = pred * y_std_t + y_mean_t
                    delta_batch = lst_pred_c - t2m_orig
                    ssrd_batch = xb[:, ssrd_feat_idx]

                    delta_centered = delta_batch - delta_batch.mean()
                    ssrd_centered = ssrd_batch - ssrd_batch.mean()
                    denom = torch.sqrt((delta_centered ** 2).sum() * (ssrd_centered ** 2).sum() + 1e-8)
                    corr = (delta_centered * ssrd_centered).sum() / denom
                    rad_penalty = torch.relu(-corr)
                    rad_losses.append(rad_penalty.item())
                    total_physics = total_physics + rad_weight * rad_penalty

                loss = loss_main + total_physics
                optimizer.zero_grad()
                loss.backward()
                xb.requires_grad_(False)

            elif is_spatial_batch:
                # Spatial regularization batch
                pred, log_var = model(xb)
                loss_main = nll_loss_fn(pred, log_var, yb, wb)

                n_sub = min(512, xb.size(0))
                idx_sub = torch.randperm(xb.size(0), device=device)[:n_sub]
                static_sub = xb[idx_sub, :n_static]
                pred_sub = pred[idx_sub]

                static_normed = F.normalize(static_sub, dim=1)
                sim_matrix = torch.mm(static_normed, static_normed.t())

                sim_mask = (sim_matrix > 0.9) & (~torch.eye(n_sub, device=device).bool())
                if sim_mask.any():
                    pred_diff = (pred_sub.unsqueeze(0) - pred_sub.unsqueeze(1)) ** 2
                    spatial_loss = (sim_matrix[sim_mask] * pred_diff[sim_mask]).mean()
                else:
                    spatial_loss = torch.tensor(0.0, device=device)

                spatial_losses.append(spatial_loss.item())
                loss = loss_main + spatial_weight * spatial_loss
                optimizer.zero_grad()
                loss.backward()

            else:
                # Standard batch with Mixup
                if mixup_alpha > 0 and epoch > warmup_epochs:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    idx = torch.randperm(xb.size(0), device=device)
                    xb_mix = lam * xb + (1 - lam) * xb[idx]
                    yb_mix = lam * yb + (1 - lam) * yb[idx]
                    wb_mix = (lam * wb + (1 - lam) * wb[idx]) if wb is not None else None
                    pred, log_var = model(xb_mix)
                    loss = nll_loss_fn(pred, log_var, yb_mix, wb_mix)
                else:
                    pred, log_var = model(xb)
                    loss = nll_loss_fn(pred, log_var, yb, wb)
                optimizer.zero_grad()
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Scheduler
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # Validation
        model.eval()
        val_losses, val_preds = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _log_var = model(xb)
                val_losses.append(val_criterion(pred, yb).item())
                val_preds.append(pred.cpu().numpy())

        val_preds_all = np.concatenate(val_preds) * y_std + y_mean
        val_rmse = np.sqrt(np.mean((y_val - val_preds_all) ** 2))

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        avg_mono = np.mean(mono_losses) if mono_losses else 0.0
        avg_rad = np.mean(rad_losses) if rad_losses else 0.0
        avg_spatial = np.mean(spatial_losses) if spatial_losses else 0.0
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_rmse"].append(val_rmse)
        history["mono_loss"].append(avg_mono)
        history["rad_loss"].append(avg_rad)
        history["spatial_loss"].append(avg_spatial)

        improved = es.step(avg_val)
        if improved:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % verbose_every == 0 or epoch == 1:
            print(f"      Epoch {epoch:3d}: train={avg_train:.5f}, val={avg_val:.5f}, "
                  f"RMSE={val_rmse:.3f}, mono={avg_mono:.5f}, rad={avg_rad:.5f}, spat={avg_spatial:.5f}")

        if es.should_stop:
            print(f"      Early stopping at epoch {epoch}")
            break

    # SWA finalization
    if use_swa and swa_model is not None and epoch >= swa_start:
        swa_bn_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train_n, dtype=torch.float32),
        )
        swa_bn_loader = DataLoader(swa_bn_ds, batch_size=batch_size, shuffle=False,
                                   num_workers=0, pin_memory=True)
        torch.optim.swa_utils.update_bn(swa_bn_loader, swa_model, device=device)
        swa_state = {k: v.cpu().clone() for k, v in swa_model.module.state_dict().items()}
        model.load_state_dict(swa_state)
        model.eval()
        swa_preds = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred, _ = model(xb)
                swa_preds.append(pred.cpu().numpy())
        swa_rmse = np.sqrt(np.mean((y_val - np.concatenate(swa_preds) * y_std + y_mean) ** 2))
        if best_state is not None:
            model.load_state_dict(best_state)
            model.eval()
            es_preds = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    pred, _ = model(xb)
                    es_preds.append(pred.cpu().numpy())
            es_rmse = np.sqrt(np.mean((y_val - np.concatenate(es_preds) * y_std + y_mean) ** 2))
            if swa_rmse < es_rmse:
                print(f"      SWA improved: RMSE {es_rmse:.4f} -> {swa_rmse:.4f}")
                best_state = swa_state
            else:
                print(f"      SWA not better: SWA={swa_rmse:.4f} vs ES={es_rmse:.4f}, keeping ES")
        else:
            best_state = swa_state

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    return model, history, best_state


def predict_phylst(model, X, y_mean, y_std, batch_size=4096, device=None):
    """Predict with PhyLST, returning (predictions_celsius, uncertainty_percent).

    Uncertainty logic:
      log_var larger → variance larger → uncertainty higher → percentage larger
      log_var smaller → variance smaller → uncertainty lower → percentage smaller
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    preds_list, unc_list = [], []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            pred_norm, log_var = model(xb)
            preds_list.append(pred_norm.cpu().numpy())
            unc_list.append(log_var.cpu().numpy())
    preds = np.concatenate(preds_list) * y_std + y_mean
    log_vars = np.concatenate(unc_list)

    # log_var → std (normalized space) → std (Celsius) → relative uncertainty → tanh → %
    variance = np.exp(log_vars)
    std_norm = np.sqrt(np.clip(variance, 1e-8, None))
    std_celsius = std_norm * y_std
    relative_unc = std_celsius / (np.abs(preds) + 1e-8)
    uncertainty_pct = np.tanh(relative_unc) * 100.0
    uncertainty_pct = np.clip(uncertainty_pct, 0, 100)

    return preds, uncertainty_pct
