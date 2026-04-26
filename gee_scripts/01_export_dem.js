/*******************************************************************************
 * DEM Export — Copernicus DEM GLO-30 → elevation, hillshade, slope, aspect
 * Study region: Greater Bay Area (YGA_BBox asset)
 * Output: 5-band GeoTIFF at 30 m (elevation, hillshade, slope, aspect_cos, aspect_sin)
 *
 * NOTE: DEM is time-invariant — export once for all years.
 ******************************************************************************/

// ====== User settings ======
var ROI_ASSET = 'projects/ee-nightsongs/assets/YGA_BBox';
var roi = ee.FeatureCollection(ROI_ASSET).geometry();

var SCALE = 30;
var EXPORT_TO_DRIVE = true;
var DRIVE_FOLDER = 'GEE_FEATURE_RASTERS';

// ====== Load Copernicus DEM GLO-30 ======
var dem = ee.ImageCollection("COPERNICUS/DEM/GLO30")
  .filterBounds(roi)
  .select('DEM')
  .mosaic()
  .clip(roi);

// ====== Derive terrain variables ======
var elevation = dem.rename('elevation');

// Slope in degrees
var slope = ee.Terrain.slope(dem).rename('slope');

// Aspect in degrees (0=N, 90=E, 180=S, 270=W)
var aspect_deg = ee.Terrain.aspect(dem);

// Convert aspect to cos/sin (circular encoding)
var aspect_rad = aspect_deg.multiply(Math.PI / 180);
var aspect_cos = aspect_rad.cos().rename('aspect_cos');
var aspect_sin = aspect_rad.sin().rename('aspect_sin');

// Hillshade (default: azimuth=315, zenith=35)
var hillshade = ee.Terrain.hillshade(dem).rename('hillshade');

// ====== Combine into 5-band image ======
var demStack = ee.Image.cat([
  elevation,    // Band 1
  hillshade,    // Band 2
  slope,        // Band 3
  aspect_cos,   // Band 4
  aspect_sin    // Band 5
]).clip(roi).toFloat();

// ====== Preview ======
Map.centerObject(roi, 9);
Map.addLayer(roi, {}, 'ROI');
Map.addLayer(elevation, {min: 0, max: 1000, palette: ['green', 'yellow', 'brown']}, 'Elevation');
Map.addLayer(slope, {min: 0, max: 45, palette: ['white', 'orange', 'red']}, 'Slope');
Map.addLayer(hillshade, {min: 100, max: 255}, 'Hillshade');

print('DEM stack bands:', demStack.bandNames());

// ====== Export ======
if (EXPORT_TO_DRIVE) {
  Export.image.toDrive({
    image: demStack,
    description: 'YGA_DEM_30m',
    fileNamePrefix: 'YGA_DEM_30m',
    folder: DRIVE_FOLDER,
    region: roi,
    scale: SCALE,
    crs: 'EPSG:4326',
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });
}

print('DEM export queued: 5 bands (elevation, hillshade, slope, aspect_cos, aspect_sin)');
