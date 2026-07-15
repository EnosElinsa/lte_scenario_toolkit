# Tests

The suite exercises the installed package and thin CLI wrappers without live
network services or full private/local data.

Coverage includes:

- schema-v2 catalog validation, scenario links, atomic updates, and incremental
  manifests;
- boundary registration from local files and mocked HTTP(S) downloads,
  including safe archive extraction, layer selection, rollback, and source
  checksums;
- mocked Earth Engine initialization, export preflight, explicit task start,
  and reproducible run records;
- disk-backed DEM shard inspection and merge, NoData/mask preservation,
  metadata checks, and boundary coverage;
- fast and full scenario validation for boundary geometry, sidecars, manifest
  containment/size/checksum drift, pending DEMs, and linked configs;
- deterministic scanning, elevation sampling, CSV/run records, and 2D/3D
  visualization using small fixtures.

Vector fixtures are small public GeoJSON or temporary Shapefiles. Raster tests
create tiny temporary GeoTIFFs with Rasterio. Earth Engine calls are mocked;
CI does not authenticate, submit exports, download Drive files, read the full
LTE point dataset, or load full-resolution city DEMs.

```powershell
python -m ruff check src scripts tests
python -m pytest -q
python -m compileall -q src/lte_scenario_toolkit scripts
```
