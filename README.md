# lte_scenario_toolkit

A reproducible research toolkit for selecting LTE base-station scenarios within U.S. city boundaries, sampling elevations from 1 m DEMs, and producing CSV data, 2D previews, 3D terrain figures, and machine-readable run records.

The project exposes a configuration-driven workflow through the `lte_scenario_toolkit` Python package. Researchers can use the installed command-line tools or the thin entry points in `scripts/`.

## Workflow

```text
Prepare the public base-station points and administrative boundaries
→ Download or place the local DEM
→ Validate the data manifest, CRS, and spatial resolution
→ Load a YAML experiment configuration
→ Scan candidate rectangles that satisfy count and spacing constraints
→ Select interactively or reproduce a selection with a fixed candidate index
→ Sample base-station elevations
→ Write CSV, PNG/EPS/HTML, and run JSON outputs
```

## Repository layout

```text
boundary_shp/                 # Boundary Shapefiles approved for public redistribution
points_shp/                   # Public LTE base stations; large components use Git LFS
dem/                          # Local DEMs; excluded from Git and Git LFS
configs/                      # Chicago and New York City experiment configurations
data/datasets.yaml            # Dataset provenance, licensing, and spatial metadata
data/manifest.json            # File sizes and SHA256 checksums
src/lte_scenario_toolkit/     # The single Python implementation package
scripts/                      # Thin CLI entry points and manifest generator
gee/newyork_1m_dem.js         # GEE Code Editor export script
tests/fixtures/               # Small public fixtures that do not require full datasets
runs/                         # Only reusable templates and concise summaries are tracked
```

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

The installation provides `lte-select-sites`, `lte-generate-figures`, and `lte-download-newyork-dem`. Equivalent `python scripts/...` commands are also available.

The distribution name and Python import name are both `lte_scenario_toolkit`:

```python
from lte_scenario_toolkit.config import load_experiment_config
from lte_scenario_toolkit.scenario import scan_rectangles
```

## Data preparation

The base-station points and boundary data are distributed with the repository. DEM rasters are much larger and remain local; see [dem/README.md](dem/README.md) for the required paths and download workflow. Dataset provenance and terms are documented in [data/README.md](data/README.md).

Check or export the New York City 1 m DEM with Earth Engine:

```powershell
python scripts/download_newyork_1m_dem.py `
  --project YOUR_EARTH_ENGINE_PROJECT_ID `
  --dry-run

python scripts/download_newyork_1m_dem.py `
  --project YOUR_EARTH_ENGINE_PROJECT_ID `
  --export
```

After downloading the exported tiles, merge them into the single GeoTIFF referenced by the configuration:

```text
dem/USGS_1M_DEM_NewYorkState_NewYork/USGS_1M_DEM_NewYorkState_NewYork.tif
```

Regenerate the complete checksum manifest after preparing or changing any input:

```powershell
python scripts/create_data_manifest.py
```

## Reproducible scenario selection

Chicago example:

```powershell
python scripts/select_sites.py --config configs/example.yaml
```

The first run can use the desktop window to choose a candidate rectangle. Record its index and use `--select-index` for formal experiments so that the selection is reproducible:

```powershell
python scripts/select_sites.py `
  --config configs/example.yaml `
  --select-index 1
```

New York City example, after preparing and merging the DEM:

```powershell
python scripts/select_sites.py `
  --config configs/newyork.yaml `
  --select-index 1
```

Common command-line overrides:

```powershell
python scripts/select_sites.py `
  --config configs/example.yaml `
  --city Chicago `
  --output-dir results/custom-run `
  --size 3000 `
  --target 30
```

Each successful run writes `run-select-sites.json` to the output directory. It records the resolved configuration, Git commit, input SHA256 checksums, software versions, and output inventory.

## Generate figures from an existing CSV

```powershell
python scripts/generate_scenario_figures.py `
  --config configs/example.yaml
```

This command reads the CSV resolved from the configuration, produces publication-style PNG/EPS figures and an interactive HTML view, and writes `run-generate-figures.json`.

## Tests and CI

Run the local checks with:

```powershell
python -m ruff check src scripts tests
python -m pytest -q
python -m compileall -q src/lte_scenario_toolkit scripts
node --check gee/newyork_1m_dem.js
```

GitHub Actions uses small vector fixtures and in-memory DEM rasters. It does not download the full datasets or access Earth Engine.

## Licensing and attribution

The source code is released under the [MIT License](LICENSE). Data does not automatically inherit the software license. The base-station and boundary datasets are published under redistribution permissions confirmed by the repository owner, and USGS 3DEP data retains its USGS attribution. Unknown original source URLs and acquisition dates remain explicitly `null` in `data/datasets.yaml` rather than being inferred.

## Known limitations

- Initial interactive candidate selection requires a desktop graphical environment; reproducible runs can use `--select-index`.
- Local scenario processing expects one GeoTIFF, so Earth Engine output tiles must be merged first.
- EPSG:3857 provides metre-based scanning at the city scale used here, but it is not an equal-area projection.
- The project currently supports source installation and GitHub distribution and is not yet published on PyPI.
