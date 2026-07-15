# Experiment configurations

Files in `configs/` describe experiment parameters, not data acquisition
provenance. The schema-v2 catalog in `data/datasets.yaml` owns provider,
license, source URL, acquisition date, CRS, and registered entrypoints. Keep
those concerns separate: changing a scan size does not rewrite dataset
metadata, and registering a new source does not silently change an experiment.

## Existing configurations

- `example.yaml` runs the Chicago scenario.
- `newyork.yaml` runs the New York City scenario.

Use them with the package entrypoints or their thin script wrappers:

```powershell
python scripts/select_sites.py --config configs/example.yaml
python scripts/generate_scenario_figures.py --config configs/example.yaml
```

Each YAML file contains `experiment`, `inputs`, `spatial`, `scan`, and
`outputs` sections. Input paths are resolved against the repository root;
outputs are resolved only when a run actually writes them. The important input
fields are `points_root`, `points_layer`, `boundary_root`, `city`, and
`dem_path`. Spatial and scan fields define the target CRS, rectangle size,
point-count tolerance, search strategy, step, candidate limit, and minimum
spacing.

## Cross-checking links

When a catalog scenario has a `config_path`, `lte-data validate` loads that
YAML and calls the same input-path resolver used by the experiment scripts with
output creation disabled. It compares the selected boundary and DEM paths to
the catalog's exact entrypoints. A malformed YAML is `config.invalid`; a
boundary or DEM mismatch is reported separately. This catches a stale city
name or moved raster before an experiment writes results.

Command-line overrides remain local to a run. They do not modify the YAML,
catalog, or generated manifest.
