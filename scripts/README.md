# Command-line wrappers

Files in this directory are thin source-tree wrappers. Reusable behavior belongs
in `src/lte_scenario_toolkit/`; installed console scripts are the public
interface.

## Data lifecycle

`manage_data.py` delegates to `lte-data`, the CLI-only scenario registration,
DEM export, ingest, and validation interface:

```powershell
lte-data scenario add <scenario-id> --boundary-source <path-or-url> `
  --provider <name> --license <terms> --redistribution-confirmed
lte-data scenario list
lte-data dem export <scenario-id> --dry-run
lte-data dem ingest <scenario-id> --tiles-dir <download-directory>
lte-data validate <scenario-id>
```

Use `--redistribution-confirmed` only after verifying that the source license
permits the normalized boundary to be tracked.

## Experiment workflow

- `select_sites.py` delegates to `lte-select-sites`. Its default interactive
  selector is the local web candidate explorer; use `--select-index N` for a
  headless deterministic choice.
- `generate_scenario_figures.py` delegates to `lte-generate-figures`. It accepts
  a completed toolkit run and supports preview/publication presets plus
  explicit PNG, EPS, and offline HTML formats.
- `benchmark_candidate_scan.py` runs the opt-in production-path scanner
  benchmark with cache and run writes disabled and prints sorted JSON metrics.
- `create_data_manifest.py` delegates to catalog manifest generation.

Examples:

```powershell
python scripts/select_sites.py --config configs/example.yaml --output-root results
python scripts/generate_scenario_figures.py --run-dir <selection-run> `
  --output-root results --format png --format html
python scripts/benchmark_candidate_scan.py --config configs/example.yaml
```

`--output-root` creates a unique run hierarchy.

The local GUI is installed as `lte-gui`; it intentionally has no source-tree
wrapper. Install the GUI extra and run `lte-gui --check` before starting it.
