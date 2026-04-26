# GEE Scripts — Auxiliary Data Export

These Google Earth Engine (GEE) scripts export the auxiliary raster data needed for PhyLST raster prediction.

## Scripts

### 1. `01_export_dem.js` — DEM (one-time)
- **Source**: Copernicus DEM GLO-30
- **Output**: 5-band GeoTIFF at 30 m
  - Band 1: `elevation` (m)
  - Band 2: `hillshade` (0–255)
  - Band 3: `slope` (degrees)
  - Band 4: `aspect_cos` (cosine of aspect)
  - Band 5: `aspect_sin` (sine of aspect)
- **Frequency**: Export once (DEM is time-invariant)
- **File**: `YGA_DEM_30m.tif`

### 2. `02_export_ndvi.js` — NDVI Annual Stats (per year)
- **Source**: Sentinel-2 SR Harmonized (B4, B8)
- **Output**: 4-band GeoTIFF at 10 m per year
  - Band 1: `ndvi_mean` (annual mean)
  - Band 2: `ndvi_p10` (10th percentile)
  - Band 3: `ndvi_p90` (90th percentile)
  - Band 4: `ndvi_amp` (p90 − p10)
- **Frequency**: One file per year (2017–2024)
- **Files**: `YGA_NDVI_2017.tif` ... `YGA_NDVI_2024.tif`

### 3. `03_export_era5.js` — ERA5-Land Monthly Meteorology (per month)
- **Source**: ECMWF/ERA5_LAND/DAILY_AGGR
- **Output**: 9-band GeoTIFF at ~11 km per month
  - Band 1: `dewpoint_2m` (°C)
  - Band 2: `soil_temp_l1` (°C)
  - Band 3: `ssrd_daily` (MJ/m²/day)
  - Band 4: `strd_daily` (MJ/m²/day)
  - Band 5: `surface_pressure` (hPa)
  - Band 6: `t2m_max` (°C)
  - Band 7: `t2m_mean` (°C)
  - Band 8: `total_precip` (mm/day)
  - Band 9: `wind_speed` (m/s)
- **Unit conversions** (applied in GEE):
  - Temperature: K → °C
  - Pressure: Pa → hPa
  - Radiation: J/m² → MJ/m²/day
  - Precipitation: m → mm
  - Wind: sqrt(u² + v²)
- **Frequency**: One file per month (2017-01 to 2024-12)
- **Files**: `YGA_ERA5_2017_01.tif` ... `YGA_ERA5_2024_12.tif`

## Study Region
- Asset: `projects/ee-nightsongs/assets/YGA_BBox`
- Region: Greater Bay Area (Guangdong), China

## Export Destination
All scripts export to Google Drive folder: `GEE_FEATURE_RASTERS`

## Usage with predict_raster.py

```bash
python predict_raster.py \
  --embedding  YGA_embedding_2020.tif \
  --dem        YGA_DEM_30m.tif \
  --ndvi       YGA_NDVI_2020.tif \
  --era5       YGA_ERA5_2020_07.tif \
  --doy 196 --year 2020 \
  --output     YGA_LST_2020_07_15.tif
```

## Band Order Alignment

The band order in each GEE export matches what `predict_raster.py` expects:

| Data | GEE Band Order | config.py Reference |
|------|---------------|-------------------|
| DEM | elevation, hillshade, slope, aspect_cos, aspect_sin | `TOPO_COLS` ✓ |
| NDVI | ndvi_mean, ndvi_p10, ndvi_p90, ndvi_amp | Reordered in code to match `NDVI_COLS` |
| ERA5 | dewpoint_2m, soil_temp_l1, ssrd_daily, strd_daily, surface_pressure, t2m_max, t2m_mean, total_precip, wind_speed | `METEO_COLS` ✓ |

**Note**: NDVI bands are exported as (mean, p10, p90, amp) but `config.py` defines `NDVI_COLS` as (amp, mean, p10, p90). The reordering is handled automatically in `predict_raster.py`.
