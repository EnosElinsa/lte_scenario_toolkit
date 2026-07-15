# Tests

The test suite covers the named package and CLI, YAML configuration, city-boundary discovery, CRS conversion, point-in-boundary filtering, deterministic rectangle scanning, spacing and point-count constraints, DEM sampling, NoData and missing-file errors, CSV fields, the data manifest, run records, 2D and 3D outputs, and workflow orchestration.

`fixtures/` contains only two small public GeoJSON files. Raster tests use in-memory GeoTIFFs created with Rasterio. Tests do not read the complete LTE dataset or 1 m DEMs and do not access Earth Engine.

```powershell
python -m ruff check src scripts tests
python -m pytest -q
```
