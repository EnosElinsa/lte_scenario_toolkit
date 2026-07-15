# DEM data

DEM rasters are external inputs. The repository's `.gitignore` excludes them,
so they are not committed to Git or Git LFS; the catalog records their
provenance and the expected local entrypoint. A scenario is still useful while
its external DEM is pending: boundary checks and catalog validation can run
before the raster is downloaded.

## Export and ingest Chicago

Set an Earth Engine Cloud Project that you control, then use the registered
scenario ID. The first command prints a plan without contacting Earth Engine;
the second performs preflight work and writes a run record; `--export` is the
explicit opt-in that starts the Earth Engine task.

```powershell
$env:EE_PROJECT = "YOUR_EARTH_ENGINE_PROJECT_ID"
lte-data dem export chicago --dry-run
lte-data dem export chicago
lte-data dem export chicago --export
# manual Drive download
lte-data dem ingest chicago --tiles-dir D:\downloads\chicago-dem
lte-data validate chicago --full-checksum
```

Download the sharded GeoTIFF files from the recorded Drive folder, then run
`dem ingest` against that directory. Ingest verifies the shard grid, merges a
single raster at the registered entrypoint, validates coverage, and refreshes
the manifest. It does not delete the downloaded tiles automatically; retain
them until the run has been checked and clean them up deliberately if desired.

## Export and ingest New York City

The New York City scenario uses the identical registered-boundary and Earth
Engine flow:

```powershell
lte-data dem export new-york-city --dry-run
lte-data dem export new-york-city
lte-data dem export new-york-city --export
# manual Drive download
lte-data dem ingest new-york-city --tiles-dir D:\downloads\new-york-city-dem
lte-data validate new-york-city --full-checksum
```

The registered boundary is the sole region of interest. The exporter derives
its geometry from the catalog entrypoint; there is no separate county or
hand-written ROI switch to keep in sync.

## Raster contract

The workflow uses the USGS 3DEP 1 m elevation collection (`USGS/3DEP/1m`),
the `elevation` band, a 1 m nominal scale, and `EPSG:3857` export metadata.
Earth Engine may produce multiple Cloud Optimized GeoTIFF shards because of
file-dimension and pixel limits. The Drive download is manual, and `dem ingest`
is the reproducible local merge step. It checks the registered CRS, resolution,
bounds, and finite coverage inside the boundary without loading a whole raster
into memory.

The catalog marks these rasters as external. A missing declared entrypoint is
reported as `dem-pending` by `lte-data validate`; it is not treated as a
repository source failure. Once the raster exists, validation checks coverage
and the manifest file sizes (and SHA256 values when `--full-checksum` is used).

## Sources and run records

The public source description is the
[USGS 3DEP 1 m Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/USGS_3DEP_1m).
Each export writes a timestamped directory under `runs/` containing the plan,
boundary checksum, task metadata, and source notes. See
[runs/README.md](../runs/README.md).
