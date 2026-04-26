"""
=============================================================================
Stage 2 — 5-Fold Cross-Validation for All Models
=============================================================================
Trains Ridge, RandomForest, XGBoost, LightGBM, MLP, ResNet1D, and PhyLST
using 5-fold CV. Saves checkpoints and collects predictions.
"""

import time
import numpy as np
import joblib
import torch

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
import xgboost as xgb
import lightgbm as lgb

from config import (
    SEED, TARGET, MODEL_NAMES, CKPT_DIR,
    RIDGE_ALPHA, RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF,
    GBDT_N_ESTIMATORS, GBDT_MAX_DEPTH, GBDT_LR, GBDT_EARLY_STOPPING,
    DL_EPOCHS_BASELINE, DL_BATCH_SIZE, DL_LR_BASELINE, DL_WD_BASELINE,
    DL_PATIENCE_BASELINE, DL_WARMUP_BASELINE,
    PHYLST_EPOCHS, PHYLST_LR, PHYLST_PATIENCE, PHYLST_WARMUP,
    PHYLST_MONO_WEIGHT, PHYLST_MONO_MARGIN, PHYLST_RAD_WEIGHT,
    PHYLST_SPATIAL_WEIGHT, PHYLST_MIXUP_ALPHA, PHYLST_SWA_START,
    PHYLST_HIDDEN_DIM, PHYLST_N_BLOCKS, PHYLST_DROPOUT,
    QUANTILE_BOOST, SUMMER_BOOST,
)
from utils import compute_metrics
from utils.weights import compute_sample_weights
from models import MLPModel, ResNet1D, PhyLST
from training.train_baselines import train_dl_model_baseline, predict_dl
from training.train_phylst import train_phylst, predict_phylst


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_cross_validation(df, FULL_FEATURES, feature_indices, verbose=True):
    """Run 5-fold cross-validation for all models.

    Parameters
    ----------
    df : DataFrame
        Cleaned DataFrame from Stage 1.
    FULL_FEATURES : list[str]
        Ordered feature column names.
    feature_indices : dict
        Physics constraint indices from Stage 1.

    Returns
    -------
    all_fold_metrics : list[dict]
        Per-fold metrics for each model.
    all_test_predictions : dict
        Concatenated test predictions per model.
    rf_importances_all : list[ndarray]
        RF feature importances per fold.
    dl_histories : dict
        DL training histories.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("STAGE 2: 5-FOLD CROSS-VALIDATION — ALL MODELS")
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
    years_all = df["year"].values
    seasons_all = df["season"].values
    lats_all = df["lat"].values
    lons_all = df["lon"].values

    # Storage
    all_fold_metrics = []
    all_test_predictions = {}
    rf_importances_all = []
    dl_histories = {"MLP": [], "ResNet1D": [], "PhyLST": []}

    for mn in MODEL_NAMES:
        all_test_predictions[mn] = {"y_true": [], "y_pred": [], "month": [], "year": [],
                                     "season": [], "lat": [], "lon": []}
    all_test_predictions["PhyLST"]["uncertainty"] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        fold_start = time.time()
        if verbose:
            print(f"\n{'─' * 60}")
            print(f"  FOLD {fold_idx + 1}/5  (train={len(train_idx)}, test={len(test_idx)})")
            print(f"{'─' * 60}")

        X_train_fold = X_all[train_idx]
        y_train_fold = y_all[train_idx]
        X_test_fold = X_all[test_idx]
        y_test_fold = y_all[test_idx]

        # Proper train/val split for early stopping
        proper_train_idx, val_idx = train_test_split(
            np.arange(len(X_train_fold)), test_size=0.15, random_state=SEED
        )
        X_proper_train = X_train_fold[proper_train_idx]
        y_proper_train = y_train_fold[proper_train_idx]
        X_val = X_train_fold[val_idx]
        y_val = y_train_fold[val_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_fold)
        X_test_sc = scaler.transform(X_test_fold)
        X_proper_train_sc = scaler.transform(X_proper_train)
        X_val_sc = scaler.transform(X_val)

        # Target stats for DL
        y_mean_fold = y_proper_train.mean()
        y_std_fold = y_proper_train.std()

        # PhyLST: t2m_mean feature stats from scaler
        t2m_feat_mean_fold = float(scaler.mean_[t2m_feat_idx])
        t2m_feat_std_fold = float(scaler.scale_[t2m_feat_idx])

        # Sample weights — ONLY for PhyLST
        train_months_fold = months_all[train_idx][proper_train_idx]
        if verbose:
            print("    Computing sample weights (for PhyLST only)...")
        sw_proper_train = compute_sample_weights(
            y_proper_train, train_months_fold,
            quantile_boost=QUANTILE_BOOST, summer_boost=SUMMER_BOOST,
        )

        # Save fold artifacts
        fold_ckpt_dir = CKPT_DIR / f"fold{fold_idx + 1}"
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, fold_ckpt_dir / "scaler.joblib")
        joblib.dump({"y_mean": float(y_mean_fold), "y_std": float(y_std_fold)},
                    fold_ckpt_dir / "target_stats.joblib")

        # Test metadata
        test_months = months_all[test_idx]
        test_years = years_all[test_idx]
        test_seasons = seasons_all[test_idx]
        test_lats = lats_all[test_idx]
        test_lons = lons_all[test_idx]

        def record_result(model_name, y_true, y_pred):
            m = compute_metrics(y_true, y_pred)
            m["model"] = model_name
            m["fold"] = fold_idx + 1
            m["n_samples"] = len(y_true)
            all_fold_metrics.append(m)
            all_test_predictions[model_name]["y_true"].append(y_true)
            all_test_predictions[model_name]["y_pred"].append(y_pred)
            all_test_predictions[model_name]["month"].append(test_months)
            all_test_predictions[model_name]["year"].append(test_years)
            all_test_predictions[model_name]["season"].append(test_seasons)
            all_test_predictions[model_name]["lat"].append(test_lats)
            all_test_predictions[model_name]["lon"].append(test_lons)
            if verbose:
                print(f"    {model_name:30s} R2={m['R2']:.4f}  RMSE={m['RMSE']:.3f}  "
                      f"MAE={m['MAE']:.3f}  Bias={m['Bias']:+.4f}")

        # ── Ridge ───────────────────────────────────────────────────
        ridge = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
        ridge.fit(X_train_sc, y_train_fold)
        record_result("Ridge", y_test_fold, ridge.predict(X_test_sc))
        joblib.dump(ridge, fold_ckpt_dir / "Ridge.joblib")

        # ── Random Forest ───────────────────────────────────────────
        rf = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            max_features=0.5, n_jobs=-1, random_state=SEED,
        )
        rf.fit(X_train_fold, y_train_fold)
        record_result("RandomForest", y_test_fold, rf.predict(X_test_fold))
        rf_importances_all.append(rf.feature_importances_)
        joblib.dump(rf, fold_ckpt_dir / "RandomForest.joblib")

        # ── XGBoost ─────────────────────────────────────────────────
        xgb_model = xgb.XGBRegressor(
            n_estimators=GBDT_N_ESTIMATORS, max_depth=GBDT_MAX_DEPTH,
            learning_rate=GBDT_LR,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
            min_child_weight=10, tree_method="hist", device="cuda",
            random_state=SEED, verbosity=0, early_stopping_rounds=GBDT_EARLY_STOPPING,
        )
        xgb_model.fit(
            X_proper_train, y_proper_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        record_result("XGBoost", y_test_fold, xgb_model.predict(X_test_fold))
        xgb_model.save_model(str(fold_ckpt_dir / "XGBoost.json"))

        # ── LightGBM ────────────────────────────────────────────────
        lgb_model = lgb.LGBMRegressor(
            n_estimators=GBDT_N_ESTIMATORS, max_depth=GBDT_MAX_DEPTH,
            learning_rate=GBDT_LR,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
            min_child_samples=20, num_leaves=63,
            random_state=SEED, verbosity=-1, n_jobs=-1,
        )
        lgb_model.fit(
            X_proper_train, y_proper_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(GBDT_EARLY_STOPPING, verbose=False),
                       lgb.log_evaluation(0)],
        )
        record_result("LightGBM", y_test_fold, lgb_model.predict(X_test_fold))
        lgb_model.booster_.save_model(str(fold_ckpt_dir / "LightGBM.txt"))

        # ── MLP ─────────────────────────────────────────────────────
        if verbose:
            print("    Training MLP (baseline)...")
        mlp = MLPModel(input_dim, hidden_dims=[256, 128], dropout=0.2)
        mlp, mlp_hist, mlp_best_state = train_dl_model_baseline(
            mlp, X_proper_train_sc, y_proper_train, X_val_sc, y_val,
            y_mean_fold, y_std_fold,
            epochs=DL_EPOCHS_BASELINE, batch_size=DL_BATCH_SIZE,
            lr=DL_LR_BASELINE, weight_decay=DL_WD_BASELINE,
            patience=DL_PATIENCE_BASELINE, verbose_every=50,
            warmup_epochs=DL_WARMUP_BASELINE, device=DEVICE,
        )
        record_result("MLP", y_test_fold,
                       predict_dl(mlp, X_test_sc, y_mean_fold, y_std_fold, device=DEVICE))
        dl_histories["MLP"].append(mlp_hist)
        if mlp_best_state is not None:
            torch.save(mlp_best_state, fold_ckpt_dir / "MLP.pt")
        del mlp
        torch.cuda.empty_cache()

        # ── ResNet1D ────────────────────────────────────────────────
        if verbose:
            print("    Training ResNet1D (baseline)...")
        resnet = ResNet1D(input_dim=input_dim, n_static=n_static,
                          hidden_dim=256, n_blocks=4, dropout=0.15)
        resnet, resnet_hist, resnet_best_state = train_dl_model_baseline(
            resnet, X_proper_train_sc, y_proper_train, X_val_sc, y_val,
            y_mean_fold, y_std_fold,
            epochs=DL_EPOCHS_BASELINE, batch_size=DL_BATCH_SIZE,
            lr=8e-4, weight_decay=DL_WD_BASELINE,
            patience=DL_PATIENCE_BASELINE, verbose_every=50,
            warmup_epochs=DL_WARMUP_BASELINE, device=DEVICE,
        )
        record_result("ResNet1D", y_test_fold,
                       predict_dl(resnet, X_test_sc, y_mean_fold, y_std_fold, device=DEVICE))
        dl_histories["ResNet1D"].append(resnet_hist)
        if resnet_best_state is not None:
            torch.save(resnet_best_state, fold_ckpt_dir / "ResNet1D.pt")
        del resnet
        torch.cuda.empty_cache()

        # ── PhyLST (PROPOSED) ───────────────────────────────────────
        if verbose:
            print("    Training PhyLST (PROPOSED — energy balance + uncertainty + expanded physics)...")
        phylst = PhyLST(
            input_dim=input_dim, n_static=n_static,
            hidden_dim=PHYLST_HIDDEN_DIM, n_blocks=PHYLST_N_BLOCKS,
            dropout=PHYLST_DROPOUT,
            t2m_feat_idx=t2m_feat_idx,
            t2m_feat_mean=t2m_feat_mean_fold,
            t2m_feat_std=t2m_feat_std_fold,
            y_mean=float(y_mean_fold),
            y_std=float(y_std_fold),
        )
        phylst, phylst_hist, phylst_best_state = train_phylst(
            phylst, X_proper_train_sc, y_proper_train, X_val_sc, y_val,
            y_mean_fold, y_std_fold,
            pos_mono_indices=pos_mono_indices,
            neg_mono_indices=neg_mono_indices,
            t2m_feat_idx=t2m_feat_idx,
            ssrd_feat_idx=ssrd_feat_idx,
            n_static=n_static,
            t2m_feat_mean=t2m_feat_mean_fold,
            t2m_feat_std=t2m_feat_std_fold,
            mono_weight=PHYLST_MONO_WEIGHT, mono_margin=PHYLST_MONO_MARGIN,
            rad_weight=PHYLST_RAD_WEIGHT, spatial_weight=PHYLST_SPATIAL_WEIGHT,
            sample_weights=sw_proper_train,
            epochs=PHYLST_EPOCHS, batch_size=DL_BATCH_SIZE,
            lr=PHYLST_LR, weight_decay=DL_WD_BASELINE,
            patience=PHYLST_PATIENCE, verbose_every=50,
            use_swa=True, swa_start_frac=PHYLST_SWA_START,
            mixup_alpha=PHYLST_MIXUP_ALPHA,
            warmup_epochs=PHYLST_WARMUP, device=DEVICE,
        )
        phylst_preds, phylst_unc = predict_phylst(
            phylst, X_test_sc, y_mean_fold, y_std_fold, device=DEVICE
        )
        record_result("PhyLST", y_test_fold, phylst_preds)
        all_test_predictions["PhyLST"]["uncertainty"].append(phylst_unc)
        dl_histories["PhyLST"].append(phylst_hist)
        if phylst_best_state is not None:
            torch.save(phylst_best_state, fold_ckpt_dir / "PhyLST.pt")
        del phylst
        torch.cuda.empty_cache()

        fold_time = time.time() - fold_start
        if verbose:
            print(f"  Fold {fold_idx + 1} completed in {fold_time:.0f}s")
            print(f"  Checkpoints saved to: {fold_ckpt_dir}")

    return all_fold_metrics, all_test_predictions, rf_importances_all, dl_histories
