# Run artifacts

`runs/` stores reproducibility records, not downloaded DEM shards or generated
research outputs. Large rasters, caches, logs, and figures remain local.

## DEM export records

Every non-dry-run `lte-data dem export <scenario-id>` call writes a uniquely
named timestamped directory such as:

```text
runs/20260716-120000-chicago-dem-export/
```

The directory is published atomically and contains:

- `RUN.md`: timestamp, scenario, Earth Engine project, Git commit, image count,
  task ID, export parameters, and software versions;
- `DATA_LAYER.md`: DEM band, scale, CRS, Cloud Optimized GeoTIFF settings,
  shard dimensions, and related task ID;
- `sources.md`: the exact registered boundary path and SHA256 plus the Earth
  Engine collection and band;
- `export-plan.json`: the machine-readable plan, boundary checksum, pixel
  estimate, task metadata, commit, and software versions.

A preflight run records `<not started>` for the task. An explicit `--export`
run records the submitted task ID. These records do not download Drive files
and do not delete local shards.

## Templates and summaries

- `templates/` contains reusable human-readable run and data-layer templates.
- `summaries/` contains concise tracked summaries of notable experiments.
- Other run directories are ignored by default and may be retained locally or
  selectively published after review.

Scenario selection and figure generation write machine-readable run JSON files
inside their output directories. Reference those records from a concise
summary instead of committing complete result trees.
