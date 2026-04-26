"""Evaluation metrics for LST prediction."""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_metrics(y_true, y_pred):
    """Compute comprehensive regression metrics.

    Returns dict with R2, RMSE, MAE, Bias, ubRMSE.
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    ubrmse = np.sqrt(np.mean((y_pred - y_true - bias) ** 2))
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "Bias": bias, "ubRMSE": ubrmse}
