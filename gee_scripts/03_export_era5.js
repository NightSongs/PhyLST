/*******************************************************************************
 * ERA5-Land Monthly Meteorological Export — 9 variables matching PhyLST training
 * Study region: Greater Bay Area (YGA_BBox asset)
 * Output: 9-band GeoTIFF per month at ~11 km (ERA5-Land native)
 *
 * Band order (matching config.py METEO_COLS):
 *   Band 1: dewpoint_2m       — monthly mean dewpoint temperature (°C)
 *   Band 2: soil_temp_l1      — monthly mean soil temperature layer 1 (°C)
 *   Band 3: ssrd_daily        — monthly mean daily SSRD (MJ/m²/day)
 *   Band 4: strd_daily        — monthly mean daily STRD (MJ/m²/day)
 *   Band 5: surface_pressure  — monthly mean surface pressure (hPa)
 *   Band 6: t2m_max           — monthly mean of daily max 2m temperature (°C)
 *   Band 7: t2m_mean          — monthly mean 2m temperature (°C)
 *   Band 8: total_precip      — monthly mean daily precipitation (mm/day)
 *   Band 9: wind_speed        — monthly mean 10m wind speed (m/s)
 *
 * Unit conversions applied:
 *   - Temperature: K → °C  (subtract 273.15)
 *   - Pressure: Pa → hPa   (divide by 100)
 *   - Radiation: J/m² → MJ/m²/day  (divide by 1e6)
 *   - Precipitation: m → mm  (multiply by 1000)
 *   - Wind speed: sqrt(u² + v²)
 *
 * Years: 2017-2025, Months: 1-12 (one file per month)
 ******************************************************************************/

// ====== User settings ======
var ROI_ASSET = 'projects/ee-nightsongs/assets/YGA_BBox';
var roi = ee.FeatureCollection(ROI_ASSET).geometry();

var START_YEAR = 2017;
var END_YEAR   = 2025;
var SCALE      = 11132;   // ERA5-Land native resolution (~0.1°)
var EXPORT_TO_DRIVE = true;
var DRIVE_FOLDER = 'GEE_FEATURE_RASTERS';

// ====== Helper: build monthly image ======
function buildMonthlyERA5(year, month) {
  year  = ee.Number(year);
  month = ee.Number(month);

  var start = ee.Date.fromYMD(year, month, 1);
  var end   = start.advance(1, 'month');
  var nDays = end.difference(start, 'day');

  var era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    .filterDate(start, end)
    .filterBounds(roi);

  // ---- Temperature variables: daily mean → monthly mean, K → °C ----
  var t2m_mean = era5.select('temperature_2m').mean()
    .subtract(273.15).rename('t2m_mean');

  var t2m_max = era5.select('temperature_2m_max').mean()
    .subtract(273.15).rename('t2m_max');

  var dewpoint = era5.select('dewpoint_temperature_2m').mean()
    .subtract(273.15).rename('dewpoint_2m');

  var soil_temp = era5.select('soil_temperature_level_1').mean()
    .subtract(273.15).rename('soil_temp_l1');

  // ---- Pressure: Pa → hPa ----
  var pressure = era5.select('surface_pressure').mean()
    .divide(100).rename('surface_pressure');

  // ---- Radiation: daily sum (J/m²) → monthly mean daily (MJ/m²/day) ----
  // sum over month / nDays → mean daily J/m², then / 1e6 → MJ/m²/day
  var ssrd = era5.select('surface_solar_radiation_downwards_sum').mean()
    .divide(1e6).rename('ssrd_daily');

  var strd = era5.select('surface_thermal_radiation_downwards_sum').mean()
    .divide(1e6).rename('strd_daily');

  // ---- Precipitation: daily sum (m) → monthly mean daily (mm/day) ----
  var precip = era5.select('total_precipitation_sum').mean()
    .multiply(1000).rename('total_precip');

  // ---- Wind speed: sqrt(u² + v²) ----
  var u10 = era5.select('u_component_of_wind_10m').mean();
  var v10 = era5.select('v_component_of_wind_10m').mean();
  var wind = u10.pow(2).add(v10.pow(2)).sqrt().rename('wind_speed');

  // ---- Stack in METEO_COLS order ----
  var monthly = ee.Image.cat([
    dewpoint,         // Band 1: dewpoint_2m
    soil_temp,        // Band 2: soil_temp_l1
    ssrd,             // Band 3: ssrd_daily
    strd,             // Band 4: strd_daily
    pressure,         // Band 5: surface_pressure
    t2m_max,          // Band 6: t2m_max
    t2m_mean,         // Band 7: t2m_mean
    precip,           // Band 8: total_precip
    wind              // Band 9: wind_speed
  ]).clip(roi).toFloat().unmask(0);

  return monthly.set({
    'year': year,
    'month': month,
    'start': start.format('YYYY-MM-dd'),
    'end': end.format('YYYY-MM-dd')
  });
}

// ====== Preview ======
Map.centerObject(roi, 9);
Map.addLayer(roi, {}, 'ROI');

var preview = buildMonthlyERA5(2020, 7);
Map.addLayer(preview.select('t2m_mean'),
  {min: 20, max: 35, palette: ['blue', 'green', 'yellow', 'red']},
  'T2m Mean Jul 2020 (°C)');
Map.addLayer(preview.select('ssrd_daily'),
  {min: 10, max: 30, palette: ['black', 'purple', 'yellow', 'white']},
  'SSRD Jul 2020 (MJ/m²/day)');

print('Preview bands:', preview.bandNames());

// ====== Export loop: year × month ======
for (var y = START_YEAR; y <= END_YEAR; y++) {
  for (var m = 1; m <= 12; m++) {
    var img  = buildMonthlyERA5(y, m);
    var mm   = (m < 10) ? '0' + m : '' + m;
    var desc = 'YGA_ERA5_' + y + '_' + mm;

    if (EXPORT_TO_DRIVE) {
      Export.image.toDrive({
        image: img,
        description: desc,
        fileNamePrefix: desc,
        folder: DRIVE_FOLDER,
        region: roi,
        scale: SCALE,
        crs: 'EPSG:4326',
        maxPixels: 1e13,
        fileFormat: 'GeoTIFF'
      });
    }
  }
}

print('Export queued: ' + START_YEAR + '-' + END_YEAR + ', 12 months/year');
print('Band order: dewpoint_2m, soil_temp_l1, ssrd_daily, strd_daily, surface_pressure, t2m_max, t2m_mean, total_precip, wind_speed');
