"""
=============================================================================
Stage 4 — Aggregate Results & Save Experiment Config
=============================================================================
"""

import json
import numpy as np
import pandas as pd

from config import (
    MODEL_NAMES, TAB_DIR, LOG_DIR, CKPT_DIR, SEED, TARGET,
    TOPO_COLS, NDVI_COLS, A_COLS, METEO_FEATURES, TEMPORAL_FEATURES,
    COORD_FEATURES, DROPPED_METEO, POS_MONO_NAMES, NEG_MONO_NAMES,
    PHYLST_MONO_WEIGHT, PHYLST_MONO_MARGIN, PHYLST_RAD_WEIGHT,
    PHYLST_SPATIAL_WEIGHT, PHYLST_HIDDEN_DIM, PHYLST_N_BLOCKS, PHYLST_DROPOUT,
)


def aggregate_results(all_fold_metrics, all_test_predictions, rf_importances_all,
                      FULL_FEATURES, feature_indices, cleaning_stats, verbose=True):
    """Aggregate cross-validation results, save tables and config.

    Returns
    -------
    summary_df : DataFrame
        Model comparison summary.
    concat_preds : dict
        Concatenated predictions per model.
    rf_importance_df : DataFrame
        Feature importance ranking.
    best_model_name : str
        Name of the best model by R².
    """
    if verbose:
        print("\n" + "=" * 70)
        print("STAGE 4: AGGREGATE RESULTS")
        print("=" * 70)

    input_dim = feature_indices["input_dim"]
    n_static = feature_indices["n_static"]

    metrics_df = pd.DataFrame(all_fold_metrics)
    metrics_df.to_csv(TAB_DIR / "metrics_all_folds.csv", index=False)

    summary_rows = []
    for model_name in MODEL_NAMES:
        model_metrics = metrics_df[metrics_df["model"] == model_name]
        row = {"model": model_name}
        for metric in ["R2", "RMSE", "MAE", "Bias", "ubRMSE"]:
            vals = model_metrics[metric].values
            row[f"{metric}_mean"] = np.mean(vals)
            row[f"{metric}_std"] = np.std(vals)
            row[f"{metric}"] = f"{np.mean(vals):.4f}\u00b1{np.std(vals):.4f}"
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TAB_DIR / "metrics_summary.csv", index=False)
    if verbose:
        print(f"Saved: {TAB_DIR / 'metrics_summary.csv'}")
        print("\n" + summary_df[["model", "R2", "RMSE", "MAE", "Bias"]].to_string(index=False))

    best_idx = summary_df["R2_mean"].idxmax()
    best_model_name = summary_df.loc[best_idx, "model"]
    if verbose:
        print(f"\nBest model: {best_model_name} (R2={summary_df.loc[best_idx, 'R2_mean']:.4f})")

    # RF feature importance
    rf_importance_df = pd.DataFrame({
        "feature": FULL_FEATURES,
        "importance_mean": np.mean(rf_importances_all, axis=0),
        "importance_std": np.std(rf_importances_all, axis=0),
    }).sort_values("importance_mean", ascending=False)
    rf_importance_df.to_csv(TAB_DIR / "rf_feature_importance.csv", index=False)

    # Concatenate predictions
    concat_preds = {}
    for mn in MODEL_NAMES:
        concat_preds[mn] = {
            "y_true": np.concatenate(all_test_predictions[mn]["y_true"]),
            "y_pred": np.concatenate(all_test_predictions[mn]["y_pred"]),
            "month": np.concatenate(all_test_predictions[mn]["month"]),
            "year": np.concatenate(all_test_predictions[mn]["year"]),
            "season": np.concatenate(all_test_predictions[mn]["season"]),
            "lat": np.concatenate(all_test_predictions[mn]["lat"]),
            "lon": np.concatenate(all_test_predictions[mn]["lon"]),
        }
    concat_preds["PhyLST"]["uncertainty"] = np.concatenate(
        all_test_predictions["PhyLST"]["uncertainty"]
    )

    # Save experiment config
    config = {
        "version": "v6",
        "seed": SEED,
        "target": TARGET,
        "n_features": input_dim,
        "n_static": n_static,
        "best_model": best_model_name,
        "feature_groups": {
            "static_topo": len(TOPO_COLS),
            "static_ndvi": len(NDVI_COLS),
            "static_spectral_raw": len(A_COLS),
            "meteorological": len(METEO_FEATURES),
            "temporal": len(TEMPORAL_FEATURES),
            "coordinates": len(COORD_FEATURES),
            "total": input_dim,
        },
        "physics_constraints": {
            "energy_balance_residual": "Model predicts delta = LST - T2m",
            "uncertainty_estimation": "Heteroscedastic NLL loss, dual-head (mean + log_var)",
            "expanded_monotonicity": {
                "positive": POS_MONO_NAMES,
                "negative": NEG_MONO_NAMES,
                "mono_weight": PHYLST_MONO_WEIGHT,
                "mono_margin": PHYLST_MONO_MARGIN,
            },
            "radiation_driven": {"rad_weight": PHYLST_RAD_WEIGHT},
            "spatial_regularization": {"spatial_weight": PHYLST_SPATIAL_WEIGHT},
        },
        "dropped_meteo": DROPPED_METEO,
        "cleaning": cleaning_stats,
    }
    with open(LOG_DIR / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"\nSaved: {LOG_DIR / 'experiment_config.json'}")

    # Checkpoint metadata
    ckpt_meta = {
        "description": "PhyLST checkpoints — PhyLST (proposed) vs standard baselines",
        "seed": SEED,
        "n_folds": 5,
        "features": FULL_FEATURES,
        "n_features": input_dim,
        "n_static": n_static,
        "best_model": best_model_name,
        "model_architectures": {
            "Ridge": {"type": "sklearn", "file": "Ridge.joblib", "needs_scaler": True},
            "RandomForest": {"type": "sklearn", "file": "RandomForest.joblib", "needs_scaler": False},
            "XGBoost": {"type": "xgboost", "file": "XGBoost.json", "needs_scaler": False},
            "LightGBM": {"type": "lightgbm", "file": "LightGBM.txt", "needs_scaler": False},
            "MLP": {"type": "pytorch", "file": "MLP.pt", "needs_scaler": True,
                    "class": "MLPModel", "kwargs": {"input_dim": input_dim, "hidden_dims": [256, 128]}},
            "ResNet1D": {"type": "pytorch", "file": "ResNet1D.pt", "needs_scaler": True,
                         "class": "ResNet1D", "kwargs": {"input_dim": input_dim, "n_static": n_static,
                                                          "hidden_dim": 256, "n_blocks": 4}},
            "PhyLST": {"type": "pytorch", "file": "PhyLST.pt", "needs_scaler": True,
                       "class": "PhyLST",
                       "kwargs": {"input_dim": input_dim, "n_static": n_static,
                                  "hidden_dim": PHYLST_HIDDEN_DIM, "n_blocks": PHYLST_N_BLOCKS,
                                  "dropout": PHYLST_DROPOUT},
                       "note": "Also requires t2m_feat_idx, t2m_feat_mean/std, y_mean/std"},
        },
    }
    for fi in range(1, 6):
        fold_dir = CKPT_DIR / f"fold{fi}"
        ckpt_meta[f"fold{fi}"] = {
            "path": str(fold_dir),
            "files": sorted([f.name for f in fold_dir.iterdir()]) if fold_dir.exists() else [],
        }
    with open(CKPT_DIR / "checkpoint_metadata.json", "w", encoding="utf-8") as f:
        json.dump(ckpt_meta, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"Saved: {CKPT_DIR / 'checkpoint_metadata.json'}")

    return summary_df, concat_preds, rf_importance_df, best_model_name
