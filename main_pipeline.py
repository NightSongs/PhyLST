"""
=============================================================================
PhyLST — Main Pipeline Entry Point
=============================================================================
Physics-Constrained Deep Learning for 10 m Land Surface Temperature
Reconstruction over Guangdong, China (2017-2024).

Usage:
    python main_pipeline.py

Stages:
    1. Data cleaning & feature design
    2. 5-fold cross-validation (7 models)
    3. Feature ablation + physics constraint ablation
    4. Aggregate results & save config
    5. Generate all publication-quality figures
=============================================================================
"""

import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# ── Setup ───────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    SEED, OUT_DIR, FIG_DIR, TAB_DIR, LOG_DIR, CKPT_DIR,
    TARGET, MODEL_NAMES, STATIC_FEATURES, A_COLS, COORD_COLS,
    ABLATION_CONFIGS,
)

# Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

PIPELINE_START = time.time()

# Create output directories
for d in [FIG_DIR, TAB_DIR, LOG_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# STAGE 1: DATA CLEANING & FEATURE DESIGN
# ══════════════════════════════════════════════════════════════════════
from pipeline.stage1_data import load_and_clean_data

df, INTERACTION_COLS, FULL_FEATURES, cleaning_stats, feat_idx = load_and_clean_data(verbose=True)

# Build feature subsets (needs FULL_FEATURES and INTERACTION_COLS)
from config import METEO_FEATURES, TEMPORAL_FEATURES
feature_subsets = {
    "meteo_only":      METEO_FEATURES + TEMPORAL_FEATURES,
    "static_only":     STATIC_FEATURES,
    "full":            FULL_FEATURES,
    "no_spectral":     [f for f in FULL_FEATURES if f not in A_COLS],
    "no_coords":       [f for f in FULL_FEATURES if f not in COORD_COLS],
    "no_interactions": [f for f in FULL_FEATURES if f not in INTERACTION_COLS],
}

# ══════════════════════════════════════════════════════════════════════
# STAGE 2: 5-FOLD CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════
from pipeline.stage2_cv import run_cross_validation

(all_fold_metrics, all_test_predictions,
 rf_importances_all, dl_histories) = run_cross_validation(
    df, FULL_FEATURES, feat_idx, verbose=True
)

# ══════════════════════════════════════════════════════════════════════
# STAGE 3: FEATURE ABLATION + PHYSICS CONSTRAINT ABLATION
# ══════════════════════════════════════════════════════════════════════
from pipeline.stage3_ablation import run_feature_ablation, run_physics_ablation

abl_summary_df = run_feature_ablation(
    df, FULL_FEATURES, INTERACTION_COLS, verbose=True
)

phys_abl_summary_df = run_physics_ablation(
    df, FULL_FEATURES, feat_idx, verbose=True
)

# ══════════════════════════════════════════════════════════════════════
# STAGE 4: AGGREGATE RESULTS
# ══════════════════════════════════════════════════════════════════════
from pipeline.stage4_aggregate import aggregate_results

(summary_df, concat_preds, rf_importance_df,
 best_model_name) = aggregate_results(
    all_fold_metrics, all_test_predictions, rf_importances_all,
    FULL_FEATURES, feat_idx, cleaning_stats,
    verbose=True
)

# ══════════════════════════════════════════════════════════════════════
# STAGE 5: PUBLICATION-QUALITY FIGURES
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STAGE 5: PUBLICATION-QUALITY FIGURES")
print("=" * 70)

from utils import compute_metrics
from config import TOPO_COLS, NDVI_COLS, METEO_COLS, TEMPORAL_COLS, COORD_COLS

from plotting.fig01_dataset_overview import plot_fig01
from plotting.fig02_correlation import plot_fig02
from plotting.fig03_model_comparison import plot_fig03
from plotting.fig04_scatter_best import plot_fig04
from plotting.fig05_feature_importance import plot_fig05
from plotting.fig06_training_curves import plot_fig06
from plotting.fig07_feature_ablation import plot_fig07
from plotting.fig08_monthly import plot_fig08
from plotting.fig09_quantile import plot_fig09
from plotting.fig10_spatial import plot_fig10
from plotting.fig11_multi_scatter import plot_fig11
from plotting.fig12_seasonal_scatter import plot_fig12
from plotting.fig13_error_cdf import plot_fig13
from plotting.fig14_taylor import plot_fig14
from plotting.fig15_residual_spatial import plot_fig15
from plotting.fig16_physics_ablation import plot_fig16
from plotting.fig17_uncertainty import plot_fig17

plot_fig01(df, TARGET, FIG_DIR)
plot_fig02(df, TOPO_COLS, NDVI_COLS, METEO_COLS, TEMPORAL_COLS, COORD_COLS, TARGET, FIG_DIR)
plot_fig03(summary_df, FIG_DIR)
plot_fig04(concat_preds, best_model_name, compute_metrics, FIG_DIR)
plot_fig05(rf_importance_df, FIG_DIR)
plot_fig06(dl_histories, FIG_DIR)
plot_fig07(abl_summary_df, feature_subsets, FIG_DIR)
plot_fig08(concat_preds, compute_metrics, FIG_DIR)
plot_fig09(concat_preds, compute_metrics, FIG_DIR)
plot_fig10(df, TARGET, FIG_DIR)
plot_fig11(concat_preds, compute_metrics, FIG_DIR)
plot_fig12(concat_preds, best_model_name, compute_metrics, FIG_DIR)
plot_fig13(concat_preds, FIG_DIR)
plot_fig14(concat_preds, FIG_DIR)
plot_fig15(concat_preds, best_model_name, FIG_DIR)
plot_fig16(phys_abl_summary_df, FIG_DIR)
plot_fig17(concat_preds, FIG_DIR)

# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
total_time = time.time() - PIPELINE_START
print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"Best model: {best_model_name}")
print(f"\nOutputs:")
print(f"  figures:      {FIG_DIR}  ({len(list(FIG_DIR.glob('*.png')))} files)")
print(f"  tables:       {TAB_DIR}  ({len(list(TAB_DIR.glob('*.csv')))} files)")
print(f"  checkpoints:  {CKPT_DIR}  (5 folds × 7 models + scalers)")
print(f"  logs:         {LOG_DIR}")
print(f"\nDone at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
