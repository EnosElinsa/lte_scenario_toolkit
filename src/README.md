# Reusable modules

`lte_scenario_toolkit/` is the installed Python package and the project's only implementation directory:

- `config.py`: YAML loading, field validation, path resolution, and flattened CLI overrides.
- `spatial.py`: boundary discovery, city selection, CRS normalisation, point-in-boundary filtering, and output paths.
- `scenario.py`: deterministic scan grids, point-count/spacing/boundary constraints, candidate validation, and fixed-index selection.
- `terrain.py`: DEM validation, elevation sampling, and valid-elevation checks.
- `io.py`: stable CSV schema, SHA256 input manifests, software versions, and run records.
- `visualization.py`: interactive selection, 2D previews, and static or interactive 3D terrain figures.
- `select_sites.py`: orchestration for candidate scanning, selection, elevation enrichment, and run recording.
- `generate_figures.py`: generation of publication figures from an existing scenario CSV.
- `newyork_dem.py`: Earth Engine export of the New York City USGS 3DEP 1 m DEM.

The `src/` directory itself is not a Python package named `src`. Add new behaviour to `lte_scenario_toolkit/` and first cover it with fixture-based tests in `tests/`.
