# PhyLST — Physics-Constrained Deep Learning for 10 m Land Surface Temperature Reconstruction

## Overview

PhyLST is a physics-constrained deep learning framework for reconstructing 10 m resolution Land Surface Temperature (LST) over Guangdong, China (2017–2024). The model integrates multiple physical priors into a dual-branch 1D ResNet architecture to achieve state-of-the-art accuracy with built-in uncertainty estimation.

### Key Innovations

1. **Energy Balance Residual Structure** — The model predicts the temperature anomaly δ = LST − T₂ₘ rather than absolute LST, encoding the physical prior that surface temperature is closely related to air temperature.
2. **Heteroscedastic Uncertainty Estimation** — A dual-head architecture outputs both the mean prediction and log-variance, trained with negative log-likelihood (NLL) loss.
3. **Expanded Monotonicity Constraints** — Gradient-based penalties enforce physically meaningful relationships: positive monotonicity for soil temperature, air temperature, and radiation; negative monotonicity for precipitation.
4. **Radiation-Driven Diurnal Constraint** — Penalizes negative correlation between the temperature anomaly and solar radiation.
5. **Spatial Regularization** — Cosine similarity on static features enforces prediction smoothness for geographically similar pixels.

## Project Structure

```
PhyLST/
├── config.py                    # All configuration constants and hyperparameters
├── main_pipeline.py             # Main entry point — runs the full 5-stage pipeline
├── predict_raster.py            # Memory-efficient raster inference (chunked, int16 output)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── gee_scripts/                 # Google Earth Engine export scripts
│   ├── README.md                # Data export documentation
│   ├── 01_export_dem.js         # Copernicus DEM → 5-band topo raster
│   ├── 02_export_ndvi.js        # Sentinel-2 → 4-band annual NDVI stats
│   └── 03_export_era5.js        # ERA5-Land → 9-band monthly meteorology
│
├── models/                      # Model architectures (one file per model)
│   ├── __init__.py              # Model registry
│   ├── mlp.py                   # MLP baseline
│   ├── resnet1d.py              # ResNet1D baseline
│   └── phylst.py                # PhyLST (our proposed model)
│
├── training/                    # Training code (separate per model type)
│   ├── __init__.py              # EarlyStopping, loss functions, LR schedule
│   ├── train_baselines.py       # Training loop for MLP and ResNet1D
│   └── train_phylst.py          # Training loop for PhyLST (physics constraints)
│
├── pipeline/                    # Pipeline stages
│   ├── __init__.py
│   ├── stage1_data.py           # Data cleaning & feature engineering
│   ├── stage2_cv.py             # 5-fold cross-validation for all 7 models
│   ├── stage3_ablation.py       # Feature ablation + physics constraint ablation
│   └── stage4_aggregate.py      # Results aggregation & experiment config
│
├── plotting/                    # All figure generation (one file per figure)
│   ├── __init__.py
│   ├── plot_config.py           # Global style: Times New Roman, font 18, colorbar
│   ├── fig01_dataset_overview.py
│   ├── fig02_correlation.py
│   ├── fig03_model_comparison.py
│   ├── fig04_scatter_best.py
│   ├── fig05_feature_importance.py
│   ├── fig06_training_curves.py
│   ├── fig07_feature_ablation.py
│   ├── fig08_monthly.py
│   ├── fig09_quantile.py
│   ├── fig10_spatial.py
│   ├── fig11_multi_scatter.py
│   ├── fig12_seasonal_scatter.py
│   ├── fig13_error_cdf.py
│   ├── fig14_taylor.py
│   ├── fig15_residual_spatial.py
│   ├── fig16_physics_ablation.py
│   └── fig17_uncertainty.py
│
└── utils/                       # Utility functions
    ├── __init__.py              # compute_metrics
    └── weights.py               # Adaptive sample weighting
```

## Models

| Model | Type | Description |
|-------|------|-------------|
| Ridge | Linear | Standard Ridge regression baseline |
| RandomForest | Ensemble | RF with 200 trees, max_depth=20 |
| XGBoost | GBDT | XGBoost with 500 estimators, max_depth=6 |
| LightGBM | GBDT | LightGBM with 500 estimators, 63 leaves |
| MLP | Deep Learning | 2-layer MLP [256, 128] baseline |
| ResNet1D | Deep Learning | 4-block dual-branch ResNet baseline |
| **PhyLST** | **Deep Learning** | **Our proposed model — 8-block ResNet with physics constraints** |

## Features (92 total)

| Group | Count | Description |
|-------|-------|-------------|
| Topographic | 5 | elevation, hillshade, slope, aspect_cos, aspect_sin |
| NDVI | 4 | NDVI_amp, NDVI_mean, NDVI_p10, NDVI_p90 |
| Spectral | 64 | Autoencoder embeddings A00–A63 |
| Meteorological | 9 | ERA5-Land variables (reduced from 11) |
| Temporal | 2 | doy_cos, doy_sin |
| Coordinates | 2 | lat, lon |
| Interactions | 6 | Seasonal × meteorological cross-terms |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
cd PhyLST
python main_pipeline.py
```

This runs all 5 stages:
1. **Data cleaning** — duplicate removal, physical constraints, per-month quantile filtering
2. **Cross-validation** — 5-fold CV for all 7 models
3. **Ablation studies** — feature ablation (LightGBM) + physics constraint ablation (PhyLST)
4. **Results aggregation** — metrics summary, feature importance, experiment config
5. **Figure generation** — 17 publication-quality figures (30+ individual panels)

### 3. Export auxiliary data from GEE

Run the three scripts in `gee_scripts/` on [Google Earth Engine Code Editor](https://code.earthengine.google.com/):
1. `01_export_dem.js` — export once (time-invariant)
2. `02_export_ndvi.js` — one file per year (2017–2024)
3. `03_export_era5.js` — one file per month (96 files total)

See `gee_scripts/README.md` for band order and unit details.

### 4. Raster prediction

```bash
python predict_raster.py \
  --embedding  YGA_embedding_2020.tif \
  --dem        YGA_DEM_10m.tif \
  --ndvi       YGA_NDVI_2020.tif \
  --era5       YGA_ERA5_2020_07.tif \
  --doy 196 --year 2020 \
  --output     YGA_LST_2020_07_15.tif \
  --strip-height 512 \
  --batch-size 8192
```

Output is a 3-band **int16** GeoTIFF (LZW compressed):
- Band 1: LST prediction → `pixel / 100.0` = °C
- Band 2: Ensemble std → `pixel / 100.0` = °C
- Band 3: Uncertainty → `pixel / 100.0` = %
- nodata = -9999

Memory-efficient features:
- Embedding data read lazily per strip (never fully loaded into RAM)
- Configurable strip height controls peak memory usage
- ERA5 reprojected once; DEM/NDVI reused across strips

## Output Structure

```
outputs_v6/
├── figures_kfold/       # All figures (PNG, 300 DPI)
├── tables_kfold/        # CSV tables (metrics, ablation, feature importance)
├── checkpoints_kfold/   # Model checkpoints (5 folds × 7 models + scalers)
└── logs_kfold/          # Experiment config JSON
```

## Plotting Conventions

All figures follow these conventions:
- **Font**: Times New Roman, size 18
- **Layout**: Each panel is saved as a separate figure (no multi-panel subplots)
- **Scatter plots**: Colorbar is always the same height as the plot area
- **Resolution**: 300 DPI
- **Style**: Top and right spines removed

## Data

- **Dataset**: `dataset/lst_dataset.v4.csv`
- **Study region**: Guangdong Province, China
- **Period**: 2017–2024
- **Target**: Landsat-derived LST (°C)

## Citation

If you use this code, please cite:

```
[Citation to be added after publication]
```

## License

This project is for academic research purposes.
