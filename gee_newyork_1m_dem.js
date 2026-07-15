// USGS 3DEP 1 m DEM for New York City, exported in EPSG:3857.
// Select Cloud project gen-lang-client-0153149292 in the Code Editor project
// selector before running this script.

var boundaryMode = 'city';  // 'city' = five NYC counties; 'county' = Manhattan.
var countyGeoid = '36061';   // New York County when boundaryMode == 'county'.
var driveFolder = 'usa-lte-base-station-data';
var filePrefix = 'USGS_1M_DEM_NewYorkState_NewYork';
var targetCrs = 'EPSG:3857';
var exportScale = 1;
var fileDimensions = 8192;

var allCounties = ee.FeatureCollection('TIGER/2018/Counties')
  .filter(ee.Filter.eq('STATEFP', '36'));

var nycCountyGeoids = ['36005', '36047', '36061', '36081', '36085'];
var boundary = boundaryMode === 'city'
  ? allCounties.filter(ee.Filter.inList('GEOID', nycCountyGeoids))
  : allCounties.filter(ee.Filter.eq('GEOID', countyGeoid));

var roi = boundary.geometry();
var demTiles = ee.ImageCollection('USGS/3DEP/1m').filterBounds(roi);
// The collection is tiled. Mosaic first, then clip once to the exact ROI.
var dem = demTiles.mosaic().select('elevation').clip(roi);

print('Boundary feature count', boundary.size());
print('Boundary names', boundary.aggregate_array('NAME'));
print('Intersecting 3DEP tile count', demTiles.size());
print('DEM projection before export', dem.select('elevation').projection());

Map.centerObject(boundary, 10);
Map.addLayer(boundary.style({color: '00FFFF', fillColor: '00000000'}), {}, 'NYC boundary');
Map.addLayer(
  dem,
  {min: 0, max: 150, palette: ['0b1f3a', '145da0', '1fa774', 'd8d174', 'f28e2b', 'ffffff']},
  'USGS 3DEP 1 m elevation'
);

// A city-scale 1 m export is large, so Earth Engine writes multiple GeoTIFF
// shards. Download all files from the Drive folder and merge locally if needed.
Export.image.toDrive({
  image: dem,
  description: filePrefix,
  folder: driveFolder,
  fileNamePrefix: filePrefix,
  region: roi,
  scale: exportScale,
  crs: targetCrs,
  maxPixels: 1e13,
  fileDimensions: fileDimensions,
  shardSize: 256,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});
