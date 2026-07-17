# lte_scenario_toolkit

A reproducible research toolkit for registering U.S. city study boundaries,
preparing 1 m elevation data, selecting LTE base-station scenarios, and
publishing traceable CSV, terrain-figure, and run artifacts. It provides both
installed CLI commands and a modern local browser interface.

## Repository layout

```text
boundary_shp/                 # Registered boundary bundles and import guidance
points_shp/                   # Public LTE base stations; large files use Git LFS
dem/                          # External local DEMs; ignored by Git
configs/                      # Legacy configs and schema-v2 experiment profiles
data/datasets.yaml            # Schema-v2 dataset and scenario catalog
data/manifest.json            # Generated file sizes and SHA256 values
src/lte_scenario_toolkit/     # Installed implementation package
scripts/                      # Thin source-tree wrappers
runs/                         # Export records, templates, and concise summaries
tests/                        # Offline fixtures and regression suite
```

The public contracts are documented in:

- [boundary_shp/README.md](boundary_shp/README.md) for boundary sources and
  atomic registration;
- [data/README.md](data/README.md) for the dataset catalog, manifest, and
  scenario readiness states;
- [dem/README.md](dem/README.md) for Earth Engine export and local shard ingest;
- [configs/README.md](configs/README.md) for versioned experiment profiles;
- [runs/README.md](runs/README.md) for selection and figure run artifacts.

## Installation

Python 3.10 or later is required. On Windows PowerShell:

```powershell
git lfs install
git lfs pull

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[gui]"
```

Use `python -m pip install -e ".[dev,gui]"` when running the test suite. The
installed commands are `lte-data`, `lte-select-sites`,
`lte-generate-figures`, and `lte-gui`.

## Local browser interface

Validate the catalog, translations, and local GUI settings without starting a
server:

```powershell
lte-gui --check
```

Start the application and open the default browser:

```powershell
lte-gui
```

The server binds to `127.0.0.1` by default and requires no authentication for
loopback-only use. Supplying a non-loopback `--host` prints a warning because
the application can read and write configured local experiment paths. The GUI
stores only language and output-root preferences in ignored
`.lte-data/gui-settings.json`.

The scenario, validation, profile, scan, DEM, generation, figure, and history
workflows are offline. The candidate map can optionally add a clearly marked
online basemap; failure or disabling that layer does not affect local DEM
selection.

The Scenarios page reports `ready`, `boundary-ready`, `dem-pending`, or
`invalid`. Only `ready` scenarios can start a scan. Fast validation checks
catalog links, boundary sidecars and geometry, manifest containment and sizes,
a usable DEM, and the linked default profile. Full Checksum additionally
streams SHA256 values.

The Configure page supports multiple schema-v2 profiles per scenario. The
catalog's `config_path` identifies the default profile. Existing legacy YAML is
readable and shown as a read-only migration preview. An explicit Save verifies
that the source has not changed, writes a canonical v2 profile, and atomically
keeps or repoints the scenario default. Profile paths and dataset IDs are
validated before a scan starts.

## Candidate selection workflow

Candidate scans use a memory-bounded row sweep; they do not materialize the
complete Cartesian grid. Both modes are deterministic for the same data,
profile, seed, and scanner version:

- `fast` stops after the configured number of valid, spaced candidates;
- `complete` evaluates the full bounded grid and retains a bounded result set.

Versioned candidate caches live under `.lte-data/cache`, independently of final
run directories. The GUI reports cache hits, supports Force Rescan, streams
progress, and cancels without returning a partial candidate set. The candidate
explorer overlays the registered boundary, base stations, candidate rectangles,
and local DEM terrain. It provides map and filmstrip layouts and locks exactly
one candidate before generation.

CLI selection uses the local web explorer by default:

```powershell
lte-select-sites --config configs/example.yaml --output-root results
```

Use the original Matplotlib selector or a reproducible headless choice when
needed:

```powershell
lte-select-sites --config configs/example.yaml --selector legacy
lte-select-sites --config configs/example.yaml --select-index 1 --output-root results
```

`--select-index` is one-based. The compatibility option `--output-dir` names an
exact final directory and fails on conflicting artifacts; `--output-root`
creates a unique `scenario/profile/timestamp-run` directory.

## Generation, figures, and history

After one candidate is confirmed, select any combination of scenario CSV, 2D
preview, terrain PNG, EPS, and self-contained offline HTML. Artifact failures
produce an explicit `partial` run rather than hiding successful outputs. Every
published `run.json` has a common identity, status, artifact, metadata, and
error envelope. Selection metadata adds the frozen profile, candidate, cache,
and input provenance; figure metadata adds the source, validated style,
provenance, and format-specific warnings.

Figures can also be derived from an existing selection run or CSV without
rescanning:

```powershell
lte-generate-figures --run-dir results/chicago/default/<run-directory> `
  --output-root results --preset publication --format png --format html

lte-generate-figures --csv legacy-scenario.csv --config configs/example.yaml `
  --rect-id 1 --output-root results
```

A legacy CSV containing multiple `rect_id` values requires an explicit
`--rect-id`; no first-row rectangle is selected silently. The GUI provides a
bounded preview before final publication. Its History page discovers valid runs
from registered roots, links derived figure runs to their parent, identifies
missing artifacts, and rebuilds entirely from files without a database.

## CLI-only data lifecycle

Scenario registration, Earth Engine export, and DEM ingest intentionally remain
CLI-only. The GUI shows copyable guidance but does not mutate the data catalog
through these operations.

### Register and inspect a scenario

```powershell
lte-data scenario add boston `
  --boundary-source data/boston-boundary.zip `
  --provider "U.S. Census Bureau" `
  --license "Public domain; source attribution retained" `
  --redistribution-confirmed

lte-data scenario list
lte-data scenario show boston
```

### Plan or submit a DEM export

```powershell
$env:EE_PROJECT = "YOUR_EARTH_ENGINE_PROJECT_ID"
lte-data dem export chicago --dry-run
lte-data dem export chicago
lte-data dem export chicago --export
```

`--dry-run` requires no Earth Engine access. The default authenticated command
records preflight without starting a task; `--export` explicitly submits it.
Download Drive shards manually, then ingest them:

```powershell
lte-data dem ingest chicago --tiles-dir downloads/chicago-dem
```

Ingest validates the tile grid, CRS, scale, mask, and registered boundary
coverage before atomic publication. It does not delete downloaded shards.

### Validate local data

```powershell
lte-data validate chicago
lte-data validate chicago --full-checksum
lte-data validate --all
```

Chicago and `new-york-city` use the same data commands.

## Opt-in scanner benchmark

The benchmark uses the production profile, preflight, spatial preparation, and
scanner path with candidate cache reads and writes disabled. It prints sorted
JSON metrics and writes no run outputs:

```powershell
python scripts/benchmark_candidate_scan.py --config configs/example.yaml
python scripts/benchmark_candidate_scan.py --config configs/newyork.yaml
```

Real-data timings are machine-specific and are intentionally not committed.

## Tests and CI

```powershell
python -m ruff check src scripts tests
python -m compileall -q src/lte_scenario_toolkit scripts
python -m pytest -q
```

Focused GUI and compatibility checks:

```powershell
python -m pytest tests/test_gui.py -q
python -m pytest tests/test_profiles.py tests/test_candidate_cache.py `
  tests/test_run_service.py tests/test_figure_service.py -q
```

CI uses small vector fixtures, temporary disk-backed rasters, and mocked Earth
Engine clients. It does not authenticate, submit exports, download Drive data,
read full city DEMs, or make GUI-core network requests.

## Licensing and attribution

Source code is released under the [MIT License](LICENSE). Data does not inherit
that license automatically. Preserve each catalog record's provider, license,
required attribution, and redistribution terms.

## Known limitations

- Earth Engine exports require a manual Drive download before local ingest.
- `EPSG:3857` supports metre-based city-scale operations but is not an
  equal-area projection.
- The project supports source installation and is not yet published on PyPI.
