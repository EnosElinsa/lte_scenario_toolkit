# Experiment configurations

YAML files are the canonical configuration interface for local scenario experiments:

- `example.yaml`: Chicago example.
- `newyork.yaml`: New York City's five-county boundary; requires a prepared and merged local DEM.

```powershell
python scripts/select_sites.py --config configs/example.yaml
python scripts/generate_scenario_figures.py --config configs/example.yaml
```

Field mapping:

| YAML field | Runtime field | Purpose |
|---|---|---|
| `experiment.name` | `experiment_name` | Run name |
| `experiment.random_seed` | `random_seed` | Deterministic random seed for `uniform` scanning |
| `inputs.points_root` | `points_root` | Base-station point root directory |
| `inputs.points_layer` | `points_layer` | Shapefile directory and layer name |
| `inputs.boundary_root` | `boundary_root` | Boundary root directory |
| `inputs.city` | `city_name` | Case-insensitive boundary directory or layer name |
| `inputs.dem_path` | `dem_path` | Path to one GeoTIFF |
| `spatial.target_crs` | `target_crs` | Analysis CRS; `EPSG:3857` is currently recommended |
| `spatial.rectangle_size_m` | `rect_size` | Candidate rectangle side length |
| `spatial.target_base_station_count` | `target_count` | Target number of base stations |
| `spatial.count_tolerance` | `tolerance` | Allowed point-count deviation |
| `scan.*` | Scan control fields | Strategy, step size, candidate limit, and centre spacing |
| `outputs.root` | `output_root` | Final output directory for the run |

`--config` is required for scenario selection and figure generation. The `--city`, `--output-dir`, `--size`, and `--target` command-line options override their YAML values.

`--select-index N` deterministically chooses the one-based candidate `N` for reproducible experiments. If omitted, the program opens the interactive selection window.
