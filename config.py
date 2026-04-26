"""
=============================================================================
PhyLST Configuration — constants for the LST reconstruction pipeline
=============================================================================
"""

from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────
PROJECT   = Path(r"C:\Python\LST_ML")
DATA_PATH = PROJECT / "dataset" / "lst_dataset.csv"
OUT_DIR   = PROJECT / "outputs"
FIG_DIR   = OUT_DIR / "figures_kfold"
TAB_DIR   = OUT_DIR / "tables_kfold"
LOG_DIR   = OUT_DIR / "logs_kfold"
CKPT_DIR  = OUT_DIR / "checkpoints_kfold"

# ── Reproducibility ─────────────────────────────────────────────────
SEED = 42

# ── Target ──────────────────────────────────────────────────────────
TARGET = "LST"

# ── Feature groups ──────────────────────────────────────────────────
A_COLS = [f"A{i:02d}" for i in range(64)]
NDVI_COLS = ["NDVI_amp", "NDVI_mean", "NDVI_p10", "NDVI_p90"]
TOPO_COLS = ["elevation", "hillshade", "slope", "aspect_cos", "aspect_sin"]

METEO_COLS = ["dewpoint_2m", "soil_temp_l1", "ssrd_daily",
              "strd_daily", "surface_pressure", "t2m_max", "t2m_mean",
              "total_precip", "wind_speed"]
DROPPED_METEO = ["skin_temp", "t2m_min"]

TEMPORAL_COLS = ["doy_cos", "doy_sin"]
COORD_COLS = ["lat", "lon"]

# Seasonal interaction feature pairs
INTERACTION_PAIRS = [
    ("doy_sin", "t2m_mean"),
    ("doy_cos", "t2m_mean"),
    ("doy_sin", "ssrd_daily"),
    ("doy_cos", "ssrd_daily"),
    ("doy_sin", "soil_temp_l1"),
    ("doy_cos", "soil_temp_l1"),
]

STATIC_FEATURES = TOPO_COLS + NDVI_COLS + A_COLS  # 73
METEO_FEATURES = METEO_COLS                        # 9
TEMPORAL_FEATURES = TEMPORAL_COLS                   # 2
COORD_FEATURES = COORD_COLS                         # 2

# ── Physics constraint feature names ────────────────────────────────
POS_MONO_NAMES = ["soil_temp_l1", "t2m_max", "t2m_mean", "ssrd_daily", "strd_daily"]
NEG_MONO_NAMES = ["total_precip"]
TEMP_FEATURE_NAMES = ["soil_temp_l1", "t2m_max", "t2m_mean"]

# ── Model names ─────────────────────────────────────────────────────
MODEL_NAMES = ["Ridge", "RandomForest", "XGBoost", "LightGBM",
               "MLP", "ResNet1D", "PhyLST"]

MODEL_SHORT = {
    "Ridge": "Ridge",
    "RandomForest": "RF",
    "XGBoost": "XGB",
    "LightGBM": "LGBM",
    "MLP": "MLP",
    "ResNet1D": "ResNet",
    "PhyLST": "PhyLST",
}

MODEL_COLORS = {
    "Ridge": "#90A4AE",
    "RandomForest": "#66BB6A",
    "XGBoost": "#FFA726",
    "LightGBM": "#AB47BC",
    "MLP": "#42A5F5",
    "ResNet1D": "#26A69A",
    "PhyLST": "#EF5350",
}

# ── Seasons ─────────────────────────────────────────────────────────
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
SEASON_COLORS = {"DJF": "#42A5F5", "MAM": "#66BB6A", "JJA": "#EF5350", "SON": "#FFA726"}
SEASON_LABELS = {"DJF": "Winter", "MAM": "Spring", "JJA": "Summer", "SON": "Autumn"}

# ── Physics constraint ablation configs ─────────────────────────────
ABLATION_CONFIGS = {
    "PhyLST-NoPhys":   {"mono_weight": 0.0,   "rad_weight": 0.0,   "spatial_weight": 0.0},
    "PhyLST-MonoOnly": {"mono_weight": 0.003, "rad_weight": 0.0,   "spatial_weight": 0.0},
    "PhyLST-Full":     {"mono_weight": 0.003, "rad_weight": 0.002, "spatial_weight": 0.001},
}

# ── Hyperparameters ─────────────────────────────────────────────────
# Sklearn baselines
RIDGE_ALPHA = 1.0
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_LEAF = 10

# GBDT baselines
GBDT_N_ESTIMATORS = 500
GBDT_MAX_DEPTH = 6
GBDT_LR = 0.05
GBDT_EARLY_STOPPING = 30

# DL baselines
DL_EPOCHS_BASELINE = 150
DL_BATCH_SIZE = 2048
DL_LR_BASELINE = 1e-3
DL_WD_BASELINE = 1e-4
DL_PATIENCE_BASELINE = 20
DL_WARMUP_BASELINE = 5

# PhyLST (our proposed method)
PHYLST_EPOCHS = 300
PHYLST_LR = 7e-4
PHYLST_PATIENCE = 35
PHYLST_WARMUP = 8
PHYLST_MONO_WEIGHT = 0.003
PHYLST_MONO_MARGIN = 0.01
PHYLST_RAD_WEIGHT = 0.002
PHYLST_SPATIAL_WEIGHT = 0.001
PHYLST_MIXUP_ALPHA = 0.1
PHYLST_SWA_START = 0.75
PHYLST_HIDDEN_DIM = 512
PHYLST_N_BLOCKS = 8
PHYLST_DROPOUT = 0.12

# ── Data cleaning ───────────────────────────────────────────────────
QUANTILE_LOWER = 0.005
QUANTILE_UPPER = 0.995

# ── Sample weighting ────────────────────────────────────────────────
QUANTILE_BOOST = 2.0
SUMMER_BOOST = 1.5
