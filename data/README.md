# Data contract

The data lifecycle has two layers:

- `datasets.yaml` is the schema-v2 catalog. It contains independent dataset
  records for points, boundaries, and DEMs, plus scenario records that link a
  boundary dataset, an optional DEM dataset, and an experiment configuration.
- `manifest.json` is generated integrity metadata. It expands the catalog into
  relative file paths, byte sizes, and SHA256 values without replacing the
  catalog's provenance fields.

## Schema-v2 catalog

Every dataset record declares a stable `dataset_id`, role, repository-relative
`path`, exact `entrypoint`, provider, license, acquisition date, CRS, spatial
resolution, and notes. Boundary records additionally declare geometry type,
feature count, and redistribution confirmation. DEM records declare the
Earth Engine collection, band, units, vertical datum, native scale, export CRS,
prefix, and Drive folder.

Each scenario has a stable ID, display name, `boundary_dataset_id`, optional
`dem_dataset_id`, and optional `config_path`. Dataset records are reusable;
scenario links decide which boundary and DEM belong together. The catalog
loader rejects duplicate IDs, unknown links, unsafe paths, and schema versions
other than 2.

Unknown source URLs and acquisition dates stay explicit `null` values. Do not
invent provenance from filenames, search results, or local timestamps.

## Readiness and validation

`lte-data scenario list` reports a catalog-derived readiness status:

- `boundary-ready`: the registered boundary exists and no DEM is declared;
- `dem-pending`: the boundary exists but the declared external DEM entrypoint
  is not present locally;
- `ready`: both registered entrypoints exist;
- `invalid`: the registered boundary entrypoint is missing.

Run structured checks with:

```powershell
lte-data validate chicago
lte-data validate chicago --full-checksum
lte-data validate --all
```

Fast validation reads the exact boundary entrypoint, required Shapefile
sidecars, catalog CRS/count/geometry contract, manifest structure, file
containment, and file sizes. `--full-checksum` additionally streams SHA256 for
the selected scenario files. A pending external DEM is a warning; a boundary
or manifest drift is an error.

If a scenario links an experiment YAML, validation loads it relative to the
repository and cross-checks the resolved boundary and DEM paths without
creating output directories.

## Manifest generation

Regenerate the manifest after adding or changing data:

```powershell
python scripts/create_data_manifest.py
python scripts/create_data_manifest.py --dataset-id boundary_chicago
```

Repeat `--dataset-id` to rehash several changed datasets while reusing valid
records for the rest. A full regeneration or full-checksum validation can be
expensive when large DEMs are available locally. DEM files remain external and
ignored by Git; only the relative file metadata and checksums are stored.
