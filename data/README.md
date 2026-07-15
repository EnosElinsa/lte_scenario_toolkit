# Data contract

## Tracked inputs

### `boundary_shp/`

The boundary data currently occupies approximately 0.8 MiB. The repository owner has confirmed public redistribution permission, so these files are stored directly in Git. Keep each Shapefile's `.shp`, `.shx`, `.dbf`, `.prj`, `.cpg`, and any available index or metadata files together.

The current workflow standardises inputs to `EPSG:3857`, WGS 84 / Pseudo-Mercator. Before using another boundary, verify its CRS, geometry type, feature count, and boundary name.

### `points_shp/`

The complete LTE base-station dataset occupies approximately 77 MiB and is approved for public release by the repository owner. Large binary components are managed with Git LFS. After cloning, run:

```powershell
git lfs pull
```

All Shapefile components must remain together. Dataset fields, provenance, and permissions follow the directory metadata and project release notes.

## External inputs

### `dem/`

Complete DEM rasters are excluded from both Git and Git LFS. Chicago and New York City are registered as separate external USGS 3DEP 1 m datasets. Each raster must be obtained from the public source or the documented Earth Engine workflow and placed in its registered local directory.

See [dem/README.md](../dem/README.md) for download, projection, and path requirements.

## Dataset terms

The MIT License applies only to repository source code and documentation. Each dataset remains subject to its original provenance, metadata, and redistribution terms. Preserve dataset attribution in papers, reports, and redistributed products.

## Reproducibility manifest

`datasets.yaml` stores dataset-level provenance, provider, licensing, acquisition date, CRS, spatial resolution, and notes. Unknown source URLs or dates are explicitly represented as `null`. Chicago and New York City have distinct DEM dataset IDs and paths.

`manifest.json` expands those declarations into local file records containing:

```text
dataset_id
source_url
provider
license
download_date
size_bytes
sha256
crs
spatial_resolution
notes
```

Regenerate it with:

```powershell
python scripts/create_data_manifest.py
```

The command streams SHA256 calculations, so it can take time when the approximately 11 GiB Chicago DEM and the New York City DEM are present locally. The rasters remain protected by `.gitignore`; only relative paths, sizes, and checksums are stored in the manifest.
