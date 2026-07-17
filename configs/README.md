# Experiment profiles

Files in `configs/` describe experiment choices. Dataset ownership remains in
`data/datasets.yaml`: boundary and DEM paths, provider, license, acquisition
date, CRS, checksums, and registered entrypoints are not repeated in a
schema-version-2 profile.

## Schema version 2

The GUI creates profiles below `configs/<scenario-id>/<profile-id>.yaml`. A
scenario may have multiple profiles; its catalog `config_path` identifies the
default. Profile IDs are stable lowercase slugs and display names are free-form
labels.

```yaml
schema_version: 2
profile:
  id: chicago-default
  display_name: Chicago default
  scenario_id: chicago
inputs:
  points_dataset_id: usa_clear_lte_base_stations
experiment:
  random_seed: 42
spatial:
  target_crs: EPSG:3857
  rectangle_size_m: 3000
  target_base_station_count: 30
  count_tolerance: 0
scan:
  mode: fast
  strategy: uniform
  step_m: 10
  max_rectangles: 100
  minimum_center_spacing_m: 3000
outputs:
  root: results
  save_csv: true
  save_preview_png: true
  save_terrain_png: true
  save_terrain_eps: true
  save_terrain_html: true
figures:
  preset: publication
  colormap: terrain
  dpi: 300
  azimuth_deg: -60.0
  elevation_deg: 30.0
  vertical_exaggeration: 1.0
  station_color: red
  station_marker_size: 20.0
  title: null
```

Relative output roots resolve from the repository root. A run creates a unique
`scenario/profile/timestamp-run` directory below that root. Candidate caches
remain in `.lte-data/cache` and can therefore be reused across output roots.

`scan.mode: fast` stops after the configured candidate limit is satisfied.
`scan.mode: complete` visits the complete bounded grid and retains a bounded
result set. Both modes are deterministic for the same registered data,
parameters, seed, and scanner version.

The GUI supports create, copy, rename, Save, default selection, and guarded
delete. Overwrite and default-changing operations require confirmation. A
default profile cannot be deleted until another same-scenario profile is
selected.

## Legacy YAML compatibility

The top-level `example.yaml` and `newyork.yaml` use the original
`experiment`/`inputs`/`spatial`/`scan`/`outputs` layout. CLI commands continue to
read them. The GUI presents their effective values as a read-only migration
preview. An explicit Save first verifies the displayed source revision, then
writes a canonical schema-version-2 profile and atomically keeps or repoints
the catalog default. Effective scan and output settings are preserved; no
legacy file changes before that confirmed Save.

Legacy YAML repeats boundary and DEM paths. Those links are checked against the
catalog so drift is reported instead of silently selecting different data.
New profiles should use dataset IDs and let the catalog remain authoritative.

## Commands

```powershell
lte-select-sites --config configs/example.yaml
lte-generate-figures --config configs/example.yaml
lte-gui
```

CLI overrides such as `--size`, `--target`, `--output-root`, and figure style
options apply only to that invocation. They do not modify the profile, catalog,
manifest, or cached source data.
