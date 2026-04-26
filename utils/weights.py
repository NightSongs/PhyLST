"""Adaptive per-sample weighting for extreme temperatures and summer months."""

import numpy as np


def compute_sample_weights(y, months, quantile_boost=2.0, summer_boost=1.5, verbose=True):
    """Upweight extreme LST quantiles (Q1/Q5) and summer months (May-Sep).

    Parameters
    ----------
    y : ndarray
        Target values (LST in °C).
    months : ndarray
        Month integers (1-12).
    quantile_boost : float
        Multiplier for extreme quantiles.
    summer_boost : float
        Extra multiplier for May-Sep samples.
    verbose : bool
        Print summary statistics.

    Returns
    -------
    weights : ndarray
        Per-sample weights, normalized so mean=1.0.
    """
    weights = np.ones(len(y), dtype=np.float32)

    edges = np.quantile(y, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    q1_mask = y < edges[1]
    q2_mask = (y >= edges[1]) & (y < edges[2])
    q4_mask = (y >= edges[3]) & (y < edges[4])
    q5_mask = y >= edges[4]

    weights[q1_mask] *= quantile_boost
    weights[q5_mask] *= quantile_boost
    weights[q2_mask] *= np.sqrt(quantile_boost)
    weights[q4_mask] *= np.sqrt(quantile_boost)

    summer_mask = np.isin(months, [5, 6, 7, 8, 9])
    weights[summer_mask] *= summer_boost

    weights /= weights.mean()

    if verbose:
        print(f"      Sample weights: min={weights.min():.2f}, max={weights.max():.2f}, "
              f"summer_frac={summer_mask.mean():.2f}, Q1/Q5_frac={(q1_mask|q5_mask).mean():.2f}")
    return weights
