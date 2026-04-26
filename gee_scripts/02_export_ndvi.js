/*******************************************************************************
 * NDVI Annual Stats Export — Sentinel-2 → NDVI_mean, NDVI_p90, NDVI_amp, NDVI_p10
 * Study region: Greater Bay Area (YGA_BBox asset)
 * Output: 4-band GeoTIFF per year at 10 m
 *   Band 1: ndvi_mean   (annual mean NDVI)
 *   Band 2: ndvi_p10    (10th percentile)
 *   Band 3: ndvi_p90    (90th percentile)
 *   Band 4: ndvi_amp    (p90 - p10, amplitude)
 *
 * Years: 2017-2025 (one file per year)
 ******************************************************************************/

// ====== User settings ======
var ROI_ASSET = 'projects/ee-nightsongs/assets/YGA_BBox';
var roi = ee.FeatureCollection(ROI_ASSET).geometry();

var YEARS = ee.List.sequence(2017, 2025).getInfo();
var SCALE = 10;
var EXPORT_TO_DRIVE = true;
var DRIVE_FOLDER = 'GEE_FEATURE_RASTERS';

// ====== Sentinel-2 SR Harmonized ======
var s2Collection = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
  .filterBounds(roi)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .select(['B4', 'B8']);

function addNDVI(img) {
  var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return ndvi.copyProperties(img, ['system:time_start']);
}

var s2ndvi = s2Collection.map(addNDVI);

// ====== Build annual NDVI stats ======
function ndviAnnualStatsImage(year) {
  year = ee.Number(year);
  var start = ee.Date.fromYMD(year, 1, 1);
  var end   = ee.Date.fromYMD(year.add(1), 1, 1);

  var ic = s2ndvi.filterDate(start, end);

  var reducer = ee.Reducer.mean().combine({
    reducer2: ee.Reducer.percentile([10, 90]),
    sharedInputs: true
  });

  var stats = ic.select('NDVI').reduce(reducer);
  // Bands: NDVI_mean, NDVI_p10, NDVI_p90

  var ndvi_mean = stats.select('NDVI_mean').rename('ndvi_mean');
  var ndvi_p10  = stats.select('NDVI_p10').rename('ndvi_p10');
  var ndvi_p90  = stats.select('NDVI_p90').rename('ndvi_p90');
  var ndvi_amp  = ndvi_p90.subtract(ndvi_p10).rename('ndvi_amp');

  // Mask pixels with no valid observations
  var count = ic.select('NDVI').count().rename('ndvi_obs_count');
  var valid = count.gt(0);

  return ee.Image.cat([ndvi_mean, ndvi_p10, ndvi_p90, ndvi_amp])
    .updateMask(valid)
    .clip(roi)
    .toFloat()
    .set({
      'year': year,
      'system:index': year.format('%d'),
      'start': start.format('YYYY-MM-dd'),
      'end': end.format('YYYY-MM-dd')
    });
}

// ====== Preview ======
Map.centerObject(roi, 9);
Map.addLayer(roi, {}, 'ROI');
var preview2020 = ndviAnnualStatsImage(2020);
Map.addLayer(preview2020.select('ndvi_mean'), {min: 0, max: 0.8, palette: ['white', 'green', 'darkgreen']}, 'NDVI Mean 2020');
Map.addLayer(preview2020.select('ndvi_amp'),  {min: 0, max: 0.6, palette: ['white', 'yellow', 'red']}, 'NDVI Amp 2020');

// ====== Export loop ======
YEARS.forEach(function(y) {
  var img  = ndviAnnualStatsImage(y);
  var desc = 'YGA_NDVI_' + y;

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
});

print('Years queued for export:', YEARS);
print('Example bands:', ndviAnnualStatsImage(2020).bandNames());
print('Band order: ndvi_mean, ndvi_p10, ndvi_p90, ndvi_amp');
