"""
=============================================================================
Stage 3 — Feature Ablation & Physics Constraint Ablation
=============================================================================
"""

import time
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
import lightgbm as lgb

from config import (
    SEED, TARGET, TAB_DIR,
    STATIC_FEATURES, METEO_FEATURES, TEMPORAL_FEATURES, COORD_FEATURES,
    A_COLS, COORD_COLS,
    GBDT_N_ESTIMATORS, GBDT_MAX_DEPTH, GBDT_LR, GBDT_EARLY_STOPPING,
    ABLATION_CONFIGS, DL_BATCH_SIZE, DL_WD_BASELINE,
    PHYLST_LR, PHYLST_MIXUP_ALPHA, PHYLST_SWA_START,
    PHYLST_HIDDEN_DIM, PHYLST_N_BLOCKS, PHYLST_DROPOUT,
    QUANTILE_BOOST, SUMMER_BOOST,
)
from utils import compute_metrics
from utils.weights import compute_sample_weights
from models import PhyLST
from training.train_phylst import train_phylst, predict_phylst


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_feature_ablation(df, FULL_FEATURES, INTERACTION_COLS, verbose=True):
    """Run feature ablation study using LightGBM with 5-fold CV.

    Returns
    -------
    abl_summary_df : DataFrame
        Summary of ablation results.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("STAGE 3: FEATURE ABLATION (LightGBM, 5-fold CV)")
        print("=" * 70)

    feature_subsets = {
        "meteo_only": METEO_FEATURES + TEMPORAL_FEATURES,
        "static_only": list(STATIC_FEATURES),
        "full": FULL_FEATURES,
        "no_spectral": [f for f in FULL_FEATURES if f not in A_COLS],
        "no_coords": [f for f in FULL_FEATURES if f not in COORD_COLS],
        "no_interactions": [f for f in FULL_FEATURES if f not in INTERACTION_COLS],
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    y_all = df[TARGET].values.astype(np.float32)

    abl_results = []
    for subset_name, subset_feats in feature_subsets.items():
        if verbose:
            print(f"\n  {subset_name} ({len(subset_feats)} features)...")
        X_sub = df[subset_feats].values.astype(np.float32)
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_sub)):
            tr_idx, vl_idx = train_test_split(
                np.arange(len(train_idx)), test_size=0.15, random_state=SEED
            )
            X_tr = X_sub[train_idx[tr_idx]]
            y_tr = y_all[train_idx[tr_idx]]
            X_vl = X_sub[train_idx[vl_idx]]
            y_vl = y_all[train_idx[vl_idx]]

            lgb_abl = lgb.LGBMRegressor(
                n_estimators=GBDT_N_ESTIMATORS, max_depth=GBDT_MAX_DEPTH,
                learning_rate=GBDT_LR,
                subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                min_child_samples=20, num_leaves=63,
                random_state=SEED, verbosity=-1, n_jobs=-1,
            )
            lgb_abl.fit(
                X_tr, y_tr,
                eval_set=[(X_vl, y_vl)],
                callbacks=[lgb.early_stopping(GBDT_EARLY_STOPPING, verbose=False),
                           lgb.log_evaluation(0)],
            )
            y_pred_abl = lgb_abl.predict(X_sub[test_idx])
            m = compute_metrics(y_all[test_idx], y_pred_abl)
            m["subset"] = subset_name
            m["n_features"] = len(subset_feats)
            m["fold"] = fold_idx + 1
            abl_results.append(m)
            if verbose:
                print(f"    Fold {fold_idx + 1}: R2={m['R2']:.4f}  RMSE={m['RMSE']:.3f}")

    abl_df = pd.DataFrame(abl_results)
    abl_summary_rows = []
    for subset_name in feature_subsets:
        sub = abl_df[abl_df["subset"] == subset_name]
        row = {"subset": subset_name, "n_features": sub["n_features"].iloc[0]}
        for metric in ["R2", "RMSE", "MAE"]:
            vals = sub[metric].values
            row[f"{metric}_mean"] = np.mean(vals)
            row[f"{metric}_std"] = np.std(vals)
            row[f"{metric}"] = f"{np.mean(vals):.4f}\u00b1{np.std(vals):.4f}"
        abl_summary_rows.append(row)
    abl_summary_df = pd.DataFrame(abl_summary_rows)
    abl_summary_df.to_csv(TAB_DIR / "feature_ablation_summary.csv", index=False)
    if verbose:
        print(f"\nSaved: {TAB_DIR / 'feature_ablation_summary.csv'}")

    return abl_summary_df


def run_physics_ablation(df, FULL_FEATURES, feature_indices, verbose=True):
    """Run physics constraint ablation study with 5-fold CV.

    Returns
    -------
    phys_abl_summary_df : DataFrame
        Summary of physics ablation results.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("STAGE 3b: PHYSICS CONSTRAINT ABLATION")
        print("=" * 70)

    input_dim = feature_indices["input_dim"]
    n_static = feature_indices["n_static"]
    pos_mono_indices = feature_indices["pos_mono_indices"]
    neg_mono_indices = feature_indices["neg_mono_indices"]
    t2m_feat_idx = feature_indices["t2m_feat_idx"]
    ssrd_feat_idx = feature_indices["ssrd_feat_idx"]

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    X_all = df[FULL_FEATURES].values.astype(np.float32)
    y_all = df[TARGET].values.astype(np.float32)
    months_all = df["month"].values

    phys_abl_results = []
    for abl_name, abl_cfg in ABLATION_CONFIGS.items():
        if verbose:
            print(f"\n  {abl_name}: mono={abl_cfg['mono_weight']}, "
                  f"rad={abl_cfg['rad_weight']}, spatial={abl_cfg['spatial_weight']}")
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_all)):
            X_train_fold = X_all[train_idx]
            y_train_fold = y_all[train_idx]
            X_test_fold = X_all[test_idx]
            y_test_fold = y_all[test_idx]

            proper_train_idx, val_idx = train_test_split(
                np.arange(len(X_train_fold)), test_size=0.15, random_state=SEED
            )
            X_proper_train = X_train_fold[proper_train_idx]
            y_proper_train = y_train_fold[proper_train_idx]
            X_val_abl = X_train_fold[val_idx]
            y_val_abl = y_train_fold[val_idx]

            scaler_abl = StandardScaler()
            X_train_sc_abl = scaler_abl.fit_transform(X_train_fold)
            X_test_sc_abl = scaler_abl.transform(X_test_fold)
            X_proper_train_sc_abl = scaler_abl.transform(X_proper_train)
            X_val_sc_abl = scaler_abl.transform(X_val_abl)

            y_mean_abl = y_proper_train.mean()
            y_std_abl = y_proper_train.std()
            t2m_feat_mean_abl = float(scaler_abl.mean_[t2m_feat_idx])
            t2m_feat_std_abl = float(scaler_abl.scale_[t2m_feat_idx])

            train_months_abl = months_all[train_idx][proper_train_idx]
            sw_abl = compute_sample_weights(y_proper_train, train_months_abl,
                                            quantile_boost=QUANTILE_BOOST,
                                            summer_boost=SUMMER_BOOST, verbose=False)

            abl_model = PhyLST(
                input_dim=input_dim, n_static=n_static,
                hidden_dim=PHYLST_HIDDEN_DIM, n_blocks=PHYLST_N_BLOCKS,
                dropout=PHYLST_DROPOUT,
                t2m_feat_idx=t2m_feat_idx,
                t2m_feat_mean=t2m_feat_mean_abl,
                t2m_feat_std=t2m_feat_std_abl,
                y_mean=float(y_mean_abl),
                y_std=float(y_std_abl),
            )
            abl_model, _, _ = train_phylst(
                abl_model, X_proper_train_sc_abl, y_proper_train,
                X_val_sc_abl, y_val_abl,
                y_mean_abl, y_std_abl,
                pos_mono_indices=pos_mono_indices,
                neg_mono_indices=neg_mono_indices,
                t2m_feat_idx=t2m_feat_idx,
                ssrd_feat_idx=ssrd_feat_idx,
                n_static=n_static,
                t2m_feat_mean=t2m_feat_mean_abl,
                t2m_feat_std=t2m_feat_std_abl,
                mono_weight=abl_cfg["mono_weight"],
                mono_margin=0.01,
                rad_weight=abl_cfg["rad_weight"],
                spatial_weight=abl_cfg["spatial_weight"],
                sample_weights=sw_abl,
                epochs=200, batch_size=DL_BATCH_SIZE,
                lr=PHYLST_LR, weight_decay=DL_WD_BASELINE,
                patience=25, verbose_every=100,
                use_swa=True, swa_start_frac=PHYLST_SWA_START,
                mixup_alpha=PHYLST_MIXUP_ALPHA,
                device=DEVICE,
            )
            abl_preds, _ = predict_phylst(abl_model, X_test_sc_abl,
                                          y_mean_abl, y_std_abl, device=DEVICE)
            m = compute_metrics(y_test_fold, abl_preds)
            m["ablation"] = abl_name
            m["fold"] = fold_idx + 1
            phys_abl_results.append(m)
            if verbose:
                print(f"    Fold {fold_idx + 1}: R2={m['R2']:.4f}  RMSE={m['RMSE']:.3f}")
            del abl_model
            torch.cuda.empty_cache()

    phys_abl_df = pd.DataFrame(phys_abl_results)
    phys_abl_summary_rows = []
    for abl_name in ABLATION_CONFIGS:
        sub = phys_abl_df[phys_abl_df["ablation"] == abl_name]
        row = {"ablation": abl_name}
        for metric in ["R2", "RMSE", "MAE"]:
            vals = sub[metric].values
            row[f"{metric}_mean"] = np.mean(vals)
            row[f"{metric}_std"] = np.std(vals)
            row[f"{metric}"] = f"{np.mean(vals):.4f}\u00b1{np.std(vals):.4f}"
        phys_abl_summary_rows.append(row)
    phys_abl_summary_df = pd.DataFrame(phys_abl_summary_rows)
    phys_abl_summary_df.to_csv(TAB_DIR / "physics_ablation_summary.csv", index=False)
    if verbose:
        print(f"\nSaved: {TAB_DIR / 'physics_ablation_summary.csv'}")

    return phys_abl_summary_df
