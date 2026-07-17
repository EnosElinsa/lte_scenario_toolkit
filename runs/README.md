# Run artifacts

`runs/` stores tracked reproducibility records for the data lifecycle. Generated
selection and figure runs normally live below a profile's local output root and
remain ignored unless deliberately summarized.

## Selection and figure runs

`--output-root <root>` and the GUI publish each operation to a unique path:

```text
<root>/<scenario-id>/<profile-id>/<UTC timestamp>-<run-id prefix>/
```

Artifacts are first written to a private staging directory and then published.
Every final directory contains `run.json` with a 32-character run ID, UTC
creation time, status, scenario/profile identity, contained relative artifact
paths, metadata, and errors. Selection metadata contains its frozen profile,
candidate, cache, scanner, and input provenance. Figure metadata contains its
source, validated style, renderer provenance, published formats, and warnings.

A run is `completed` when every requested artifact succeeds. If independent
formats fail, successful files are retained and the record is explicitly
`partial`. Selection runs contain exactly one locked candidate. Figure runs
derived from a selection record their parent run ID and path.

Candidate caches and bounded GUI previews are not final artifacts. They live
under ignored `.lte-data/cache`, so choosing a fresh run directory does not
discard reusable scan work.

The GUI History page discovers valid `run.json` files from the repository
results directory and user-registered output roots. It rebuilds from files,
links parents and derived figures, reports malformed or missing artifacts as
diagnostics, and never requires a database. Legacy `run-select-sites.json` and
`run-generate-figures.json` records are normalized in memory without rewriting
their directories.

The compatibility option `--output-dir <exact-directory>` preserves older
automation. It refuses conflicting artifact names and writes the corresponding
legacy operation record beside the outputs.

## DEM export records

Every non-dry-run `lte-data dem export <scenario-id>` call writes a unique
timestamped directory such as:

```text
runs/20260716-120000-chicago-dem-export/
```

It contains:

- `RUN.md`: timestamp, scenario, Earth Engine project, Git commit, task ID,
  export parameters, and software versions;
- `DATA_LAYER.md`: DEM band, scale, CRS, Cloud Optimized GeoTIFF settings,
  shards, and related task ID;
- `sources.md`: registered boundary path/checksum and Earth Engine source;
- `export-plan.json`: machine-readable plan, checksums, estimates, task
  metadata, commit, and software versions.

A preflight record uses `<not started>` for the task. Only an explicit
`--export` submits work. Records do not contain Earth Engine tokens,
application credentials, or a copy of the process environment.

## Templates and summaries

- `templates/` contains reusable human-readable run and data-layer templates.
- `summaries/` contains concise tracked summaries of notable experiments.
- Other run directories, caches, logs, and figures are ignored by default.

Reference reviewed local records from a concise summary instead of committing
complete result trees or machine-specific benchmark timings.
