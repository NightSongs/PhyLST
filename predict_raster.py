"""
=============================================================================
PhyLST Raster Prediction — Memory-Efficient Chunked Inference
=============================================================================
Predicts Land Surface Temperature (LST) for the full study region using
trained PhyLST models with 5-fold ensemble.

Input data (3 separate sources):
  1. Embedding raster:  64-band GeoTIFF (A00~A63) at 10 m — REFERENCE GRID
  2. DEM raster:        5-band GeoTIFF at 10 m
       Band 1: elevation, Band 2: hillshade, Band 3: slope,
       Band 4: aspect_cos, Band 5: aspect_sin
  3. NDVI raster:       4-band GeoTIFF at 10 m (per year)
       Band 1: ndvi_mean, Band 2: ndvi_p10, Band 3: ndvi_p90, Band 4: ndvi_amp
  4. ERA5 raster:       9-band GeoTIFF at ~11 km (per month)
       Band 1: dewpoint_2m, Band 2: soil_temp_l1, Band 3: ssrd_daily,
       Band 4: strd_daily, Band 5: surface_pressure, Band 6: t2m_max,
       Band 7: t2m_mean, Band 8: total_precip, Band 9: wind_speed

Output: 3-band int16 GeoTIFF (LZW compressed)
  Band 1: LST prediction  (value / 100 = °C,  nodata = -9999)
  Band 2: Ensemble std     (value / 100 = °C,  nodata = -9999)
  Band 3: Uncertainty      (value / 100 = %,   nodata = -9999)

Usage:
  python predict_raster.py \\
    --embedding  YGA_embedding_2020.tif \\
    --dem        YGA_DEM_10m.tif \\
    --ndvi       YGA_NDVI_2020.tif \\
    --era5       YGA_ERA5_2020_07.tif \\
    --doy 196 --year 2020 \\
    --output     YGA_LST_2020_07_15.tif

Study region: Greater Bay Area, China | Period: 2017-2024
=============================================================================
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.fill import fillnodata
except ImportError:
    print("ERROR: rasterio is required. Install with: pip install rasterio")
    sys.exit(1)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CKPT_DIR, INTERACTION_PAIRS,
    PHYLST_HIDDEN_DIM, PHYLST_N_BLOCKS, PHYLST_DROPOUT,
)
from models.phylst import PhyLST

# ── Output encoding constants ───────────────────────────────────────
SCALE_FACTOR = 100          # LST (°C) × 100 → int16
NODATA_INT16 = -9999
OUTPUT_DTYPE = "int16"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# Geo utilities
# =====================================================================
def reproject_to_match(src_arr, src_profile, dst_profile,
                       resampling=Resampling.bilinear):
    """Reproject a 2D array to match the destination grid."""
    src = np.where(np.isfinite(src_arr), src_arr, -9999.0).astype(np.float32)
    dst_h, dst_w = dst_profile["height"], dst_profile["width"]
    dst = np.full((dst_h, dst_w), -9999.0, dtype=np.float32)
    reproject(
        source=src, destination=dst,
        src_transform=src_profile["transform"], src_crs=src_profile["crs"],
        dst_transform=dst_profile["transform"], dst_crs=dst_profile["crs"],
        src_nodata=-9999.0, dst_nodata=-9999.0,
        resampling=resampling,
    )
    return np.where(dst == -9999.0, np.nan, dst).astype(np.float32)


def fill_nans_for_valid(arr, valid_mask, max_search=100):
    """Fill NaN gaps within valid_mask using nearest-neighbor interpolation."""
    out = arr.astype(np.float32, copy=True)
    need_fill = valid_mask & (~np.isfinite(out))
    if not np.any(need_fill):
        return out
    filled = fillnodata(out, mask=np.isfinite(out).astype(np.uint8),
                        max_search_distance=max_search, smoothing_iterations=0)
    out[need_fill] = filled[need_fill]
    return out


def build_coord_grids(transform, H, W):
    """Build lat/lon grids from rasterio transform (EPSG:4326 assumed)."""
    cols, rows = np.meshgrid(np.arange(W, dtype=np.float64),
                             np.arange(H, dtype=np.float64))
    lon = transform.c + (cols + 0.5) * transform.a
    lat = transform.f + (rows + 0.5) * transform.e
    return lat.astype(np.float32), lon.astype(np.float32)


# =====================================================================
# Feature assembly (row-strip version for memory efficiency)
# =====================================================================
METEO_NAMES = ["dewpoint_2m", "soil_temp_l1", "ssrd_daily", "strd_daily",
               "surface_pressure", "t2m_max", "t2m_mean", "total_precip", "wind_speed"]


def build_features_for_strip(emb_strip, dem_strip, ndvi_strip, era5_strip,
                             lat_strip, lon_strip, doy):
    """Build the 92-feature array for a horizontal strip.

    Parameters
    ----------
    emb_strip  : (64, strip_h, W)
    dem_strip  : (5, strip_h, W)   — elevation, hillshade, slope, aspect_cos, aspect_sin
    ndvi_strip : (4, strip_h, W)   — ndvi_mean, ndvi_p10, ndvi_p90, ndvi_amp
    era5_strip : (9, strip_h, W)   — 9 meteo variables (already reprojected)
    lat_strip  : (strip_h, W)
    lon_strip  : (strip_h, W)
    doy        : int

    Returns
    -------
    X : (N_valid, 92) float32
    valid_flat : (strip_h * W,) bool
    """
    sh, W = emb_strip.shape[1], emb_strip.shape[2]
    N = sh * W

    # Flatten all inputs: (bands, h, w) → (N, bands)
    emb_flat  = emb_strip.reshape(64, N).T     # (N, 64)
    dem_flat  = dem_strip.reshape(5, N).T      # (N, 5)
    ndvi_flat = ndvi_strip.reshape(4, N).T     # (N, 4)
    era5_flat = era5_strip.reshape(9, N).T     # (N, 9)
    lat_flat  = lat_strip.reshape(N)
    lon_flat  = lon_strip.reshape(N)

    # Temporal encoding
    doy_sin = np.float32(np.sin(2 * np.pi * doy / 365.25))
    doy_cos = np.float32(np.cos(2 * np.pi * doy / 365.25))

    # NDVI band order from GEE: mean, p10, p90, amp
    # config.py NDVI_COLS order: NDVI_amp, NDVI_mean, NDVI_p10, NDVI_p90
    # Reorder: amp(3), mean(0), p10(1), p90(2)
    ndvi_reordered = ndvi_flat[:, [3, 0, 1, 2]]  # amp, mean, p10, p90

    # DEM band order from GEE: elevation, hillshade, slope, aspect_cos, aspect_sin
    # config.py TOPO_COLS: elevation, hillshade, slope, aspect_cos, aspect_sin — same order ✓

    # Static features: topo(5) + NDVI(4) + spectral(64) = 73
    static = np.column_stack([dem_flat, ndvi_reordered, emb_flat])  # (N, 73)

    # Meteo: 9 features (already in correct order from GEE export)
    meteo = era5_flat  # (N, 9)

    # Temporal: doy_cos, doy_sin
    temporal = np.column_stack([
        np.full(N, doy_cos, dtype=np.float32),
        np.full(N, doy_sin, dtype=np.float32),
    ])

    # Coordinates
    coords = np.column_stack([lat_flat, lon_flat])

    # Interaction features (6)
    interactions = []
    for col_a, col_b in INTERACTION_PAIRS:
        a_val = doy_sin if col_a == "doy_sin" else doy_cos
        b_idx = METEO_NAMES.index(col_b)
        interactions.append(np.float32(a_val) * era5_flat[:, b_idx])
    interactions = np.column_stack(interactions)  # (N, 6)

    # Full: static(73) + meteo(9) + temporal(2) + coords(2) + interactions(6) = 92
    X = np.column_stack([static, meteo, temporal, coords, interactions])

    # Valid mask: no NaN in any feature
    valid_flat = ~np.any(np.isnan(X), axis=1)

    return X, valid_flat


# =====================================================================
# Model loading & prediction
# =====================================================================
def load_fold_models(n_folds=5):
    """Pre-load all fold scalers, stats, and model weights."""
    folds = []
    for fold in range(1, n_folds + 1):
        fold_dir = CKPT_DIR / f"fold{fold}"
        scaler = joblib.load(fold_dir / "scaler.joblib")
        stats = joblib.load(fold_dir / "target_stats.joblib")
        state = torch.load(fold_dir / "PhyLST.pt", map_location="cpu",
                           weights_only=True)
        folds.append({
            "scaler": scaler,
            "y_mean": stats["y_mean"],
            "y_std": stats["y_std"],
            "state_dict": state,
        })
        print(f"  Loaded fold {fold} checkpoint")
    return folds


def predict_chunk(X_valid, fold_data, batch_size=4096):
    """Run 5-fold ensemble on a chunk of valid pixels.

    Returns
    -------
    mean_pred, std_pred, mean_unc : ndarray (N,) float32
    """
    n_folds = len(fold_data)
    all_preds = np.empty((n_folds, X_valid.shape[0]), dtype=np.float32)
    all_uncs  = np.empty((n_folds, X_valid.shape[0]), dtype=np.float32)

    for fi, fd in enumerate(fold_data):
        scaler = fd["scaler"]
        y_mean, y_std = fd["y_mean"], fd["y_std"]

        X_sc = scaler.transform(X_valid)

        t2m_feat_idx = 82  # position of t2m_mean in FULL_FEATURES (73+6=79... let me compute)
        # static(73) + meteo index of t2m_mean = 73 + 6 = 79
        # Actually: meteo order is dewpoint(0), soil(1), ssrd(2), strd(3), pressure(4), t2m_max(5), t2m_mean(6), precip(7), wind(8)
        # So t2m_mean is at static(73) + 6 = index 79
        t2m_feat_idx = 79
        t2m_feat_mean = float(scaler.mean_[t2m_feat_idx])
        t2m_feat_std  = float(scaler.scale_[t2m_feat_idx])

        model = PhyLST(
            input_dim=X_valid.shape[1], n_static=73,
            hidden_dim=PHYLST_HIDDEN_DIM, n_blocks=PHYLST_N_BLOCKS,
            dropout=PHYLST_DROPOUT,
            t2m_feat_idx=t2m_feat_idx,
            t2m_feat_mean=t2m_feat_mean, t2m_feat_std=t2m_feat_std,
            y_mean=y_mean, y_std=y_std,
        )
        model.load_state_dict(fd["state_dict"])
        model = model.to(DEVICE).eval()

        ds = TensorDataset(torch.tensor(X_sc, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            pin_memory=(DEVICE.type == "cuda"), num_workers=0)

        preds_list, uncs_list = [], []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(DEVICE, non_blocking=True)
                pred_norm, log_var = model(xb)
                preds_list.append(pred_norm.cpu().numpy())
                uncs_list.append(log_var.cpu().numpy())

        preds = np.concatenate(preds_list) * y_std + y_mean
        log_vars = np.concatenate(uncs_list)
        variance = np.exp(log_vars)
        std_celsius = np.sqrt(np.clip(variance, 1e-8, None)) * y_std
        unc_pct = np.tanh(std_celsius / (np.abs(preds) + 1e-8)) * 100.0

        all_preds[fi] = preds
        all_uncs[fi]  = unc_pct

        del model, X_sc, ds, loader
        torch.cuda.empty_cache()

    mean_pred = all_preds.mean(axis=0)
    std_pred  = all_preds.std(axis=0)
    mean_unc  = all_uncs.mean(axis=0)
    return mean_pred, std_pred, mean_unc


# =====================================================================
# int16 encoding
# =====================================================================
def encode_to_int16(arr_float, scale=SCALE_FACTOR):
    """Encode float array to int16: value = round(float × scale)."""
    result = np.full(arr_float.shape, NODATA_INT16, dtype=np.int16)
    valid = np.isfinite(arr_float)
    if np.any(valid):
        scaled = np.round(arr_float[valid] * scale).astype(np.float64)
        np.clip(scaled, -9998, 32767, out=scaled)
        result[valid] = scaled.astype(np.int16)
    return result


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="PhyLST Raster Prediction — memory-efficient chunked inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output encoding:
  Band 1: LST (°C)          → pixel / 100.0
  Band 2: Ensemble std (°C) → pixel / 100.0
  Band 3: Uncertainty (%%)    → pixel / 100.0
  nodata = -9999
        """)
    parser.add_argument("--embedding", required=True,
                        help="64-band embedding GeoTIFF (10 m)")
    parser.add_argument("--dem", required=True,
                        help="5-band DEM GeoTIFF (elevation, hillshade, slope, aspect_cos, aspect_sin)")
    parser.add_argument("--ndvi", required=True,
                        help="4-band NDVI GeoTIFF (ndvi_mean, ndvi_p10, ndvi_p90, ndvi_amp)")
    parser.add_argument("--era5", required=True,
                        help="9-band ERA5 monthly GeoTIFF (see band order in header)")
    parser.add_argument("--doy", type=int, required=True,
                        help="Day of year (1-366)")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--output", default="lst_prediction.tif",
                        help="Output GeoTIFF path")
    parser.add_argument("--batch-size", type=int, default=8192,
                        help="GPU batch size for model inference")
    parser.add_argument("--strip-height", type=int, default=512,
                        help="Number of rows per processing strip (controls peak RAM)")
    parser.add_argument("--max-fill-distance", type=int, default=100,
                        help="Max pixel distance for NaN gap-filling in reprojected data")
    args = parser.parse_args()

    t0 = time.time()
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  VRAM: {gpu_mem:.1f} GB")

    # ── Load embedding metadata (don't read data yet) ───────────────
    print(f"\nOpening embedding: {args.embedding}")
    emb_ds = rasterio.open(args.embedding)
    emb_profile = emb_ds.profile.copy()
    H, W = emb_ds.height, emb_ds.width
    n_bands_emb = emb_ds.count
    print(f"  Size: {H} × {W} ({H * W:,} pixels), {n_bands_emb} bands")
    print(f"  Estimated embedding size: {H * W * n_bands_emb * 4 / 1e9:.2f} GB (float32)")

    # ── Load DEM (same grid as embedding, small) ────────────────────
    print(f"\nLoading DEM: {args.dem}")
    with rasterio.open(args.dem) as ds:
        dem_profile = ds.profile.copy()
        # Check if DEM needs reprojection
        if (ds.height == H and ds.width == W and
                ds.transform == emb_ds.transform):
            dem_data = ds.read().astype(np.float32)
            print(f"  DEM grid matches embedding — no reprojection needed")
        else:
            print(f"  DEM grid differs ({ds.height}×{ds.width}) — will reproject per strip")
            dem_data = None  # will reproject per strip
            dem_raw = ds.read().astype(np.float32)

    # ── Load NDVI (same grid as embedding, small) ───────────────────
    print(f"\nLoading NDVI: {args.ndvi}")
    with rasterio.open(args.ndvi) as ds:
        ndvi_profile = ds.profile.copy()
        if (ds.height == H and ds.width == W and
                ds.transform == emb_ds.transform):
            ndvi_data = ds.read().astype(np.float32)
            print(f"  NDVI grid matches embedding — no reprojection needed")
        else:
            print(f"  NDVI grid differs ({ds.height}×{ds.width}) — will reproject per strip")
            ndvi_data = None
            ndvi_raw = ds.read().astype(np.float32)

    # ── Load ERA5 (coarse grid, always needs reprojection) ──────────
    print(f"\nLoading ERA5: {args.era5}")
    with rasterio.open(args.era5) as ds:
        era5_profile = ds.profile.copy()
        era5_raw = ds.read().astype(np.float32)
        print(f"  ERA5 grid: {ds.height}×{ds.width} ({ds.count} bands)")

    # ── Reproject ERA5 to full embedding grid ───────────────────────
    print(f"\nReprojecting ERA5 to embedding grid ({H}×{W})...")
    era5_reproj = np.empty((9, H, W), dtype=np.float32)
    for b in range(9):
        era5_reproj[b] = reproject_to_match(
            era5_raw[b], era5_profile, emb_profile,
            resampling=Resampling.bilinear
        )
    del era5_raw
    print(f"  ERA5 reprojection done")

    # ── Reproject DEM/NDVI if needed ────────────────────────────────
    if dem_data is None:
        print(f"Reprojecting DEM to embedding grid...")
        dem_data = np.empty((5, H, W), dtype=np.float32)
        for b in range(5):
            dem_data[b] = reproject_to_match(
                dem_raw[b], dem_profile, emb_profile,
                resampling=Resampling.bilinear
            )
        del dem_raw
        print(f"  DEM reprojection done")

    if ndvi_data is None:
        print(f"Reprojecting NDVI to embedding grid...")
        ndvi_data = np.empty((4, H, W), dtype=np.float32)
        for b in range(4):
            ndvi_data[b] = reproject_to_match(
                ndvi_raw[b], ndvi_profile, emb_profile,
                resampling=Resampling.bilinear
            )
        del ndvi_raw
        print(f"  NDVI reprojection done")

    # ── Build coordinate grids ──────────────────────────────────────
    lat_grid, lon_grid = build_coord_grids(emb_ds.transform, H, W)

    # ── Build valid mask from DEM (embedding read lazily per strip) ─
    # Use DEM elevation as primary valid mask (non-NaN)
    valid_mask_global = np.isfinite(dem_data[0])
    n_valid_total = int(valid_mask_global.sum())
    print(f"\nGlobal valid pixels (from DEM): {n_valid_total:,} / {H * W:,} "
          f"({n_valid_total / max(H * W, 1) * 100:.1f}%)")

    # ── Fill NaN gaps in auxiliary data within valid mask ────────────
    print(f"Filling NaN gaps in auxiliary data (max_distance={args.max_fill_distance})...")
    for b in range(9):
        era5_reproj[b] = fill_nans_for_valid(era5_reproj[b], valid_mask_global,
                                              args.max_fill_distance)
    for b in range(4):
        ndvi_data[b] = fill_nans_for_valid(ndvi_data[b], valid_mask_global,
                                            args.max_fill_distance)
    print(f"  Gap filling done")

    # ── Load model checkpoints ──────────────────────────────────────
    print(f"\nLoading PhyLST model checkpoints...")
    fold_data = load_fold_models(n_folds=5)

    # ── Prepare output file ─────────────────────────────────────────
    out_profile = emb_profile.copy()
    out_profile.update(
        count=3,
        dtype=OUTPUT_DTYPE,
        nodata=NODATA_INT16,
        compress="LZW",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_ds = rasterio.open(out_path, "w", **out_profile)

    # ── Strip-based processing ──────────────────────────────────────
    strip_h = args.strip_height
    n_strips = (H + strip_h - 1) // strip_h
    print(f"\nProcessing {n_strips} strips (strip_height={strip_h}, "
          f"DOY={args.doy}, year={args.year})...")
    print(f"  Estimated peak RAM per strip: "
          f"{strip_h * W * (64 + 5 + 4 + 9 + 2) * 4 / 1e9:.2f} GB")

    total_predicted = 0
    t_start = time.time()

    for strip_idx in range(n_strips):
        row_start = strip_idx * strip_h
        row_end = min(row_start + strip_h, H)
        sh = row_end - row_start

        # Read embedding strip lazily from disk
        window = rasterio.windows.Window(0, row_start, W, sh)
        emb_strip = emb_ds.read(window=window).astype(np.float32)  # (64, sh, W)

        # Slice auxiliary data
        dem_strip  = dem_data[:, row_start:row_end, :]
        ndvi_strip = ndvi_data[:, row_start:row_end, :]
        era5_strip = era5_reproj[:, row_start:row_end, :]
        lat_strip  = lat_grid[row_start:row_end, :]
        lon_strip  = lon_grid[row_start:row_end, :]

        # Build features
        X_strip, valid_flat = build_features_for_strip(
            emb_strip, dem_strip, ndvi_strip, era5_strip,
            lat_strip, lon_strip, args.doy
        )
        del emb_strip  # free embedding memory immediately

        n_valid = int(valid_flat.sum())
        if n_valid == 0:
            # Write nodata for this strip
            nodata_strip = np.full((sh, W), NODATA_INT16, dtype=np.int16)
            for band_i in range(1, 4):
                out_ds.write(nodata_strip, band_i, window=window)
            if (strip_idx + 1) % 10 == 0 or strip_idx == n_strips - 1:
                print(f"  Strip {strip_idx + 1}/{n_strips}: 0 valid pixels (skipped)")
            continue

        X_valid = X_strip[valid_flat].astype(np.float32)
        del X_strip

        # Predict
        mean_pred, std_pred, mean_unc = predict_chunk(
            X_valid, fold_data, batch_size=args.batch_size
        )
        del X_valid
        total_predicted += n_valid

        # Reconstruct 2D arrays and encode to int16
        for band_i, arr_1d in enumerate([mean_pred, std_pred, mean_unc], start=1):
            full_strip = np.full(sh * W, np.nan, dtype=np.float32)
            full_strip[valid_flat] = arr_1d
            full_strip = full_strip.reshape(sh, W)
            encoded = encode_to_int16(full_strip)
            out_ds.write(encoded, band_i, window=window)

        del mean_pred, std_pred, mean_unc
        gc.collect()

        elapsed = time.time() - t_start
        rate = total_predicted / max(elapsed, 1)
        remaining_pixels = n_valid_total - total_predicted
        eta = remaining_pixels / max(rate, 1)

        if (strip_idx + 1) % 5 == 0 or strip_idx == n_strips - 1:
            print(f"  Strip {strip_idx + 1}/{n_strips}: "
                  f"{n_valid:,} px | total: {total_predicted:,}/{n_valid_total:,} | "
                  f"{rate:.0f} px/s | ETA: {eta / 60:.1f} min")

    # ── Finalize ────────────────────────────────────────────────────
    out_ds.set_band_description(1, "LST_prediction_celsius_x100")
    out_ds.set_band_description(2, "LST_ensemble_std_celsius_x100")
    out_ds.set_band_description(3, "LST_uncertainty_percent_x100")
    out_ds.close()
    emb_ds.close()

    total_time = time.time() - t0
    file_size = out_path.stat().st_size / 1e6

    print(f"\n{'=' * 60}")
    print(f"PREDICTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Output: {out_path}")
    print(f"  Size: {file_size:.1f} MB")
    print(f"  Pixels predicted: {total_predicted:,}")
    print(f"  Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"  Throughput: {total_predicted / max(total_time, 1):.0f} px/s")
    print(f"\n  Decoding:")
    print(f"    LST (°C) = pixel_value / {SCALE_FACTOR}")
    print(f"    nodata = {NODATA_INT16}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
