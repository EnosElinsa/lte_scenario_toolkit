# Reusable modules

`lte_scenario_toolkit/` is the installed Python package and the project's only
implementation directory.

## Data lifecycle

- `data_catalog.py`: schema-v2 catalog loading, indexed dataset/scenario links,
  safe relative paths, atomic catalog saves, and incremental manifests.
- `boundary_data.py`: local or HTTP(S) source staging, safe archive handling,
  layer selection, polygon normalization, source checksums, and atomic scenario
  registration.
- `dem_data.py`: generic registered-scenario Earth Engine export plans,
  explicit task submission, timestamped run records, disk-backed shard merge,
  and DEM coverage validation.
- `data_validation.py`: fast size-based and optional full-checksum validation of
  boundaries, manifest records, DEM coverage, and linked configuration paths.
- `data_cli.py`: the `lte-data` command hierarchy and stable exit behavior.

## Experiment workflow

- `config.py` loads experiment YAML and resolves repository-relative values.
- `spatial.py` discovers input layers and prepares projected spatial data.
- `scenario.py` scans and validates candidate rectangles deterministically.
- `terrain.py` validates and samples elevation values.
- `io.py` writes stable tabular outputs, manifests, and run records.
- `visualization.py` renders interactive selection and 2D/3D figures.
- `select_sites.py` and `generate_figures.py` orchestrate installed commands.

The `src/` directory itself is not a Python package named `src`. Put new logic
inside `lte_scenario_toolkit/` and cover it with fixture-based tests in
`tests/`; keep files in `scripts/` as delegation-only wrappers.
