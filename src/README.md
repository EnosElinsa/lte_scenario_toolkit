# Reusable modules

`lte_scenario_toolkit/` is the installed package and the project's only
implementation directory. CLI commands and the GUI call the same service
layer; wrappers and pages do not duplicate scan, generation, or publication
logic.

## Data lifecycle

- `data_catalog.py`: catalog loading, safe paths, indexed
  dataset/scenario links, atomic saves, and incremental manifests.
- `boundary_data.py`: local or HTTP(S) staging, safe archive handling, polygon
  normalization, provenance, and atomic scenario registration.
- `dem_data.py`: registered-boundary Earth Engine plans, explicit submission,
  run records, disk-backed shard merge, and DEM coverage checks.
- `data_validation.py`: fast size-based and optional full-checksum validation.
- `data_cli.py`: the `lte-data` command hierarchy and stable exit behavior.

Registration, Earth Engine export, and DEM ingest remain CLI-only.

## Experiment services

- `profiles.py`: current profile models, validation, discovery, CRUD, and
  defaults.
- `candidate_scanner.py`: deterministic memory-bounded row sweep, exact
  boundary/count/spacing checks, fast/complete modes, progress, and
  cancellation.
- `candidate_cache.py`: content-addressed candidate cache below
  `.lte-data/cache`.
- `selection_service.py`: catalog/profile preflight, cached scanning, DEM
  statistics, one-candidate locking, and partial-safe artifact publication.
- `map_assets.py`: bounded local DEM overlays and hillshade cache assets.
- `figure_service.py`: validated completed-run loading, bounded previews,
  configurable final figures, and provenance.
- `run_service.py`: unique staging/publication directories, immutable run
  discovery, parent links, and diagnostics.
- `jobs.py`: shared single-job coordination, progress events, and cancellation.
- `web_selector.py`: blocking CLI adapter for the local browser selector.
- `benchmark.py`: opt-in production-path scanner measurements without cache or
  output writes.

`select_sites.py` and `generate_figures.py` are installed CLI adapters around
these services.

## Local GUI

`gui/app.py` provides `lte-gui`, loopback-first startup, local settings, and
allowlisted file routes. `gui/pages/` contains thin NiceGUI pages for scenario
status, validation, profile management, candidate exploration, generation,
figures, and file-based history. `gui/i18n.py` owns matching English and Chinese
strings; `gui/assets/` contains local CSS.

NiceGUI is imported only when the application is created or started, so core
package imports and CLI workflows do not require the GUI extra. GUI core
operations are offline; the optional candidate basemap is isolated from local
DEM rendering.

The `src/` directory itself is not a Python package named `src`. Add new
behavior inside `lte_scenario_toolkit/`, cover it with fixture-based tests, and
keep `scripts/` delegation-only.
