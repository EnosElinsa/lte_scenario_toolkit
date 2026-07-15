# lte_scenario_toolkit

A reproducible research toolkit for registering U.S. city study boundaries,
preparing 1 m elevation data, selecting LTE base-station scenarios, and
producing traceable CSV, figure, and run artifacts.

The installed package exposes one generic data lifecycle command, `lte-data`,
plus experiment commands for scenario selection and figure generation. A
registered boundary is the sole region of interest used for a scenario's DEM
export and local coverage checks.

## Repository layout

```text
boundary_shp/                 # Registered boundary bundles and import guidance
points_shp/                   # Public LTE base stations; large files use Git LFS
dem/                          # External local DEMs; ignored by Git
configs/                      # Experiment parameters, separate from provenance
data/datasets.yaml            # Schema-v2 dataset and scenario catalog
data/manifest.json            # Generated file sizes and SHA256 values
src/lte_scenario_toolkit/     # Installed implementation package
scripts/                      # Thin source-tree wrappers
runs/                         # Export records, templates, and concise summaries
tests/                        # Offline fixtures and regression suite
```

The public contracts are documented in:

- [boundary_shp/README.md](boundary_shp/README.md) for supported boundary
  sources and atomic registration;
- [data/README.md](data/README.md) for the schema-v2 catalog, manifest, and
  readiness states;
- [dem/README.md](dem/README.md) for generic Earth Engine export and shard
  ingest;
- [configs/README.md](configs/README.md) for experiment YAML;
- [runs/README.md](runs/README.md) for reproducibility artifacts.

## Installation

Python 3.10 or later is required. On Windows PowerShell:

```powershell
git lfs install
git lfs pull

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Installation provides `lte-data`, `lte-select-sites`, and
`lte-generate-figures`. `lte-data` is the public extension point for scenario
data; `scripts/manage_data.py` is its thin source-tree wrapper.

## Generic data lifecycle

### 1. Register a boundary

Register a local or HTTP(S) boundary source with explicit provenance and
redistribution terms:

```powershell
lte-data scenario add boston `
  --boundary-source data/boston-boundary.zip `
  --provider "U.S. Census Bureau" `
  --license "Public domain; source attribution retained" `
  --redistribution-confirmed

lte-data scenario list
lte-data scenario show boston
```

The registration pipeline validates and normalizes the polygon, installs its
Shapefile components atomically, declares a pending external DEM, and updates
the catalog and manifest. Use `--layer` for a multi-layer GeoPackage or
archive.

### 2. Plan or submit a DEM export

```powershell
$env:EE_PROJECT = "YOUR_EARTH_ENGINE_PROJECT_ID"
lte-data dem export chicago --dry-run
lte-data dem export chicago
lte-data dem export chicago --export
```

`--dry-run` resolves and prints the plan without Earth Engine access. The
default command performs authenticated preflight and writes a run record
without starting a task. `--export` explicitly submits the Drive export. The
registered boundary entrypoint is always the exact export ROI; there is no
second city-specific ROI definition.

### 3. Download and ingest shards

Download the recorded Drive shards manually, then merge and register them:

```powershell
lte-data dem ingest chicago --tiles-dir downloads/chicago-dem
```

Ingest validates the tile grid, CRS, scale, mask, and boundary coverage before
atomically publishing the registered GeoTIFF and refreshing its manifest
record. It does not delete the downloaded shards.

### 4. Validate scenario data

```powershell
lte-data validate chicago
lte-data validate chicago --full-checksum
lte-data validate --all
```

Fast validation checks catalog links, the exact boundary and Shapefile
sidecars, geometry metadata, manifest structure/containment/sizes, an available
DEM, and any linked configuration. Full mode additionally streams SHA256 for
the selected files. A missing declared external DEM is a valid `dem-pending`
warning; boundary or manifest drift fails validation.

Chicago and `new-york-city` use the same export, ingest, and validation
commands. See [dem/README.md](dem/README.md) for complete examples.

## Run an experiment

Chicago example:

```powershell
python scripts/select_sites.py --config configs/example.yaml --select-index 1
python scripts/generate_scenario_figures.py --config configs/example.yaml
```

New York City example:

```powershell
python scripts/select_sites.py --config configs/newyork.yaml --select-index 1
python scripts/generate_scenario_figures.py --config configs/newyork.yaml
```

`--select-index` records a stable one-based candidate choice. Without it, the
selector opens an interactive desktop window. Successful commands write
machine-readable run JSON files beside their outputs.

## Tests and CI

```powershell
python -m ruff check src scripts tests
python -m pytest -q
python -m compileall -q src/lte_scenario_toolkit scripts
```

CI uses small vector fixtures, temporary disk-backed rasters, and mocked Earth
Engine clients. It does not authenticate, submit exports, download Drive data,
or read full local DEMs.

## Licensing and attribution

Source code is released under the [MIT License](LICENSE). Data does not inherit
that license automatically. Preserve each catalog record's provider, license,
and required attribution. Unknown source URLs and acquisition dates remain
`null`; they are never inferred.

## Known limitations

- Interactive candidate selection requires a desktop environment; fixed
  candidate indexes support headless reproducibility.
- Earth Engine exports are sharded and require a manual Drive download before
  local ingest.
- `EPSG:3857` supports metre-based city-scale operations but is not an
  equal-area projection.
- The project supports source installation and is not yet published on PyPI.
