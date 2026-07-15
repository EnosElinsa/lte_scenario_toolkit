# DEM data

Complete DEM rasters are not committed to Git. This directory is reserved for locally downloaded raster files, and `.gitignore` excludes the large data products.

The data registry treats the Chicago and New York City DEMs as two independent external datasets:

- `usgs_3dep_1m_dem_chicago`
- `usgs_3dep_1m_dem_new_york_city`

## Chicago

The existing Chicago raster uses the USGS National Map 3DEP 1 m DEM collection and currently occupies approximately 10.96 GiB locally. Place it at:

```text
dem/USGS_1M_DEM_Chicago/USGS_1M_DEM_Chicago.tif
```

## New York City

Dataset: `USGS/3DEP/1m`

- Band: `elevation`
- Units: metres
- Vertical datum: NAVD88
- Source imagery: USGS 3DEP 1 m ImageCollection
- Output CRS: `EPSG:3857`
- Export destination: tiled GeoTIFF files in Google Drive

Export with Python:

```powershell
python scripts/download_newyork_1m_dem.py `
  --project YOUR_EARTH_ENGINE_PROJECT_ID `
  --dry-run

python scripts/download_newyork_1m_dem.py `
  --project YOUR_EARTH_ENGINE_PROJECT_ID `
  --export
```

Alternatively, paste `gee/newyork_1m_dem.js` into the Earth Engine Code Editor and start the export from the `Tasks` panel.

The default region is the union of New York City's five counties: Bronx, Kings, New York, Queens, and Richmond. To export Manhattan/New York County only, pass `--boundary-mode county --county-geoid 36061` to the Python command or set `boundaryMode` to `'county'` in the JavaScript file.

A city-scale 1 m export normally produces multiple tiles. Download and merge them with QGIS, Rasterio, or another geospatial tool into:

```text
dem/USGS_1M_DEM_NewYorkState_NewYork/USGS_1M_DEM_NewYorkState_NewYork.tif
```

Do not commit the tiles or merged raster to GitHub.

## Official sources

- [USGS 3DEP 1 m Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/USGS_3DEP_1m)
- [Earth Engine `Export.image.toDrive`](https://developers.google.com/earth-engine/apidocs/export-image-todrive)
