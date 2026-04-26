"""
=============================================================================
Stage 1 — Data Cleaning & Feature Design
=============================================================================
Loads the raw CSV, removes duplicates and outliers, creates interaction
features, and returns a clean DataFrame with all feature columns.
"""

import numpy as np
import pandas as pd

from config import (
    DATA_PATH, TARGET, A_COLS, NDVI_COLS, TOPO_COLS, METEO_COLS,
    TEMPORAL_COLS, COORD_COLS, INTERACTION_PAIRS, DROPPED_METEO,
    STATIC_FEATURES, METEO_FEATURES, TEMPORAL_FEATURES, COORD_FEATURES,
    POS_MONO_NAMES, NEG_MONO_NAMES, TEMP_FEATURE_NAMES,
    QUANTILE_LOWER, QUANTILE_UPPER,
)


def build_full_features(interaction_cols):
    """Construct the full feature list in canonical order."""
    return STATIC_FEATURES + METEO_FEATURES + TEMPORAL_FEATURES + COORD_FEATURES + interaction_cols


def load_and_clean_data(verbose=True):
    """Load dataset, clean outliers, create interaction features.

    Returns
    -------
    df : DataFrame
        Cleaned DataFrame with all features.
    INTERACTION_COLS : list[str]
        Names of the interaction features.
    FULL_FEATURES : list[str]
        Ordered list of all feature column names.
    cleaning_stats : dict
        Statistics about the cleaning process.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("STAGE 1: DATA CLEANING & FEATURE DESIGN")
        print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    if verbose:
        print(f"Raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Remove exact duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    n_dup_removed = n_before - len(df)
    if verbose:
        print(f"After removing exact duplicates: {len(df)} rows (removed {n_dup_removed})")

    # Stage 1: Physical constraint
    n_before = len(df)
    df = df[df[TARGET] >= 0].copy()
    n_phys_removed = n_before - len(df)
    if verbose:
        print(f"Stage 1 — Physical (LST >= 0): {len(df)} rows (removed {n_phys_removed})")

    # Stage 2: Per-month quantile filtering
    df["month"] = df["acq_date"].dt.month
    n_before = len(df)
    monthly_bounds = {}
    keep_mask = pd.Series(True, index=df.index)
    for m in range(1, 13):
        month_idx = df[df["month"] == m].index
        month_lst = df.loc[month_idx, TARGET]
        lo = month_lst.quantile(QUANTILE_LOWER)
        hi = month_lst.quantile(QUANTILE_UPPER)
        monthly_bounds[m] = (float(lo), float(hi))
        month_mask = (month_lst >= lo) & (month_lst <= hi)
        keep_mask.loc[month_idx] = month_mask
        n_removed_m = (~month_mask).sum()
        if verbose:
            print(f"  Month {m:2d}: [{lo:5.1f}, {hi:5.1f}]  removed {n_removed_m}")

    df = df[keep_mask].copy()
    n_quantile_removed = n_before - len(df)
    n_total_removed = n_phys_removed + n_quantile_removed
    if verbose:
        print(f"Stage 2 — Per-month [{QUANTILE_LOWER*100:.1f}%, {QUANTILE_UPPER*100:.1f}%]: "
              f"{len(df)} rows (removed {n_quantile_removed})")
        print(f"Total outliers removed: {n_total_removed} "
              f"({n_total_removed/(n_total_removed+len(df))*100:.2f}%)")

    # Seasonal interaction features
    INTERACTION_COLS = []
    for col_a, col_b in INTERACTION_PAIRS:
        name = f"IX_{col_a}_{col_b}"
        df[name] = df[col_a] * df[col_b]
        INTERACTION_COLS.append(name)
    if verbose:
        print(f"  Seasonal interaction features: {len(INTERACTION_COLS)}")

    FULL_FEATURES = build_full_features(INTERACTION_COLS)
    input_dim = len(FULL_FEATURES)
    n_static = len(STATIC_FEATURES)

    if verbose:
        print(f"\nFeature groups:")
        print(f"  Static (topo+NDVI+spectral_64): {len(STATIC_FEATURES)}")
        print(f"  Meteorological (reduced):       {len(METEO_FEATURES)} (dropped {DROPPED_METEO})")
        print(f"  Temporal encoding:              {len(TEMPORAL_FEATURES)}")
        print(f"  Coordinates:                    {len(COORD_FEATURES)}")
        print(f"  Seasonal interactions:          {len(INTERACTION_COLS)}")
        print(f"  TOTAL:                          {input_dim}")

    # Derived columns
    df["year"] = df["acq_date"].dt.year
    df["season"] = df["month"].map({12: "DJF", 1: "DJF", 2: "DJF",
                                     3: "MAM", 4: "MAM", 5: "MAM",
                                     6: "JJA", 7: "JJA", 8: "JJA",
                                     9: "SON", 10: "SON", 11: "SON"})

    # Physics constraint feature indices
    pos_mono_indices = [FULL_FEATURES.index(f) for f in POS_MONO_NAMES]
    neg_mono_indices = [FULL_FEATURES.index(f) for f in NEG_MONO_NAMES]
    t2m_feat_idx = FULL_FEATURES.index("t2m_mean")
    ssrd_feat_idx = FULL_FEATURES.index("ssrd_daily")

    if verbose:
        print(f"  Positive monotonicity indices: {pos_mono_indices} ({POS_MONO_NAMES})")
        print(f"  Negative monotonicity indices: {neg_mono_indices} ({NEG_MONO_NAMES})")
        print(f"  t2m_mean feature index: {t2m_feat_idx}")
        print(f"  ssrd_daily feature index: {ssrd_feat_idx}")

    cleaning_stats = {
        "n_dup_removed": n_dup_removed,
        "n_phys_removed": n_phys_removed,
        "n_quantile_removed": n_quantile_removed,
        "n_total_removed": n_total_removed,
        "monthly_bounds": monthly_bounds,
    }

    feature_indices = {
        "pos_mono_indices": pos_mono_indices,
        "neg_mono_indices": neg_mono_indices,
        "t2m_feat_idx": t2m_feat_idx,
        "ssrd_feat_idx": ssrd_feat_idx,
        "n_static": n_static,
        "input_dim": input_dim,
    }

    return df, INTERACTION_COLS, FULL_FEATURES, cleaning_stats, feature_indices
