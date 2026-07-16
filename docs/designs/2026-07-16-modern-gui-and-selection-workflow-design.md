# Modern GUI and Scenario Selection Workflow Design

Date: 2026-07-16
Status: Approved in conversation

## Goal

Add a modern, local web GUI to `lte_scenario_toolkit` while strengthening the
scenario-selection and figure-generation workflows that already power the CLI.
Users must be able to choose a registered city, manage named experiment
profiles, scan for parameter-compliant candidate rectangles, inspect candidates
over local DEM terrain, select exactly one rectangle, and generate reproducible
artifacts in a chosen output root.

The GUI is an additional interface, not a second implementation. CLI commands
and GUI pages must call the same package-level services and produce compatible
configuration, cache, output, and run-record artifacts.

## Confirmed Product Decisions

- The GUI is a local web application launched with `lte-gui` and opened in the
  user's browser.
- The GUI uses NiceGUI for application state and pages, Leaflet for the map, and
  Plotly for interactive three-dimensional previews.
- End users install Python dependencies only. Node.js is not required.
- Core workflows work offline. An online basemap is an optional layer and must
  never be required for scenario selection or figure generation.
- English is the default interface language. Users can switch to Chinese.
  Configuration keys, file formats, logs, and run records remain English.
- All catalog scenarios are visible with their readiness status. Scenarios that
  are not `ready` cannot start a selection run.
- A city can own multiple named YAML experiment profiles and one default
  profile. The GUI can create, copy, rename, edit, set the default, and delete
  profiles under `configs/`.
- One run uses one city, one profile, and one set of parameters. Parameter
  sweeps are outside the design.
- The system generates parameter-compliant candidates but does not score or
  rank their scientific suitability. The user selects one candidate manually.
- Multi-selection and arbitrary map-drawn rectangles are outside the design.
- The default candidate view is a full map. Users can switch to a map plus a
  bottom candidate filmstrip without losing map or selection state.
- The default DEM style combines elevation color with hillshade. Elevation-only
  and hillshade-only modes and opacity controls remain available.
- Candidate details include station count, center coordinates, minimum,
  maximum, and mean elevation, and elevation range. These values are displayed
  for manual judgment and never affect ordering.
- Candidate scanning supports a fast mode and a complete mode. Fast mode is the
  default.
- A completed scan can be reused automatically from cache. Users can force a
  rescan.
- A selected candidate is locked first, then reviewed on a separate generation
  page. No output is written merely by clicking a candidate.
- Figure generation is available both as the next step of a selection run and
  as an independent page that opens an existing run directory or compatible
  CSV.
- Run history is file-based. No application database is introduced.
- The GUI exposes scenario status, details, and data validation. Scenario
  registration, Earth Engine DEM export, and DEM shard ingest remain CLI-only.

## Scope

The first release includes:

- shared package-level workflow services;
- experiment-profile schema and profile management;
- a memory-bounded, deterministic candidate scanner;
- versioned candidate, DEM-display, and candidate-statistics caches;
- an enhanced web candidate selector and a legacy Matplotlib fallback;
- local DEM visualization, candidate inspection, and single selection;
- selection export and configurable two- and three-dimensional figures;
- low-resolution figure previews and high-resolution final exports;
- file-based run history and derived figure runs;
- scenario status, details, fast validation, and optional full-checksum
  validation;
- English and Chinese interface strings;
- backward-compatible CLI entrypoints and legacy artifact readers.

## Non-Goals

The first release does not include:

- automatic candidate quality scoring or recommendation;
- selecting or exporting multiple candidate rectangles in one run;
- drawing, dragging, or resizing a rectangle manually;
- parameter sweeps or concurrent selection runs;
- editing the dataset catalog's provenance records in the GUI;
- registering scenarios or submitting and ingesting DEM exports in the GUI;
- a server deployment, multi-user access, user accounts, or authentication;
- a database-backed configuration or run-history service;
- mobile-phone layout support;
- exposing arbitrary Matplotlib or Plotly keyword arguments.

## Existing Problems

The current commands are packaged, but the workflow remains concentrated in
large `main` functions. The CLI loads data, scans, opens an interactive window,
samples the DEM, writes outputs, renders figures, and creates run records in one
control path. This makes progress reporting, cancellation, GUI reuse, and
focused testing difficult.

The current scan constructs and randomly permutes the complete Cartesian grid.
With the committed defaults, the Chicago bounding box contains approximately
23.2 million scan positions and New York City contains approximately 34.5
million. Materializing those positions consumes substantial memory before any
candidate is evaluated. Each position then performs a point-count mask in
Python.

The current Matplotlib selector displays boundaries, stations, and candidate
rectangles, but not DEM terrain. It supports only a blocking single-selection
window and exposes little candidate detail.

The current experiment YAML repeats boundary and DEM paths that are already
owned by `data/datasets.yaml`. The linked-config checks catch drift, but the
duplication makes GUI profile management unnecessarily fragile.

The figure workflow assumes `EPSG:3857` for a loaded CSV, silently reads
rectangle metadata from the first row, and exposes few style controls. Figure
rendering and file writing are coupled.

Finally, candidate caches live beside final outputs. Choosing a new unique run
directory prevents useful cache reuse, while reusing an old output directory
risks overwriting artifacts.

## Design Principles

- Keep `data/datasets.yaml` authoritative for registered data and provenance.
- Keep experiment profiles responsible for experiment parameters, not physical
  boundary or DEM paths.
- Give CLI and GUI entrypoints one shared implementation.
- Keep compute and file services independent of NiceGUI.
- Keep one active CPU-intensive job per local GUI process.
- Validate before creating final output directories.
- Use atomic writes for profiles, caches, run records, and final artifacts.
- Never silently overwrite an existing run or profile.
- Preserve exact geometric and station-count constraints.
- Record algorithm and schema versions wherever ordering or interpretation can
  change.
- Treat UI previews as display aids. Use registered source data for scientific
  outputs.
- Surface partial failures explicitly instead of reducing them to console
  warnings.

## Architecture

The package is divided into four layers:

```text
interfaces
  lte-gui | lte-select-sites | lte-generate-figures | lte-data validate
      |
      v
application services
  profiles | scenarios | candidate jobs | selection | figures | runs | maps
      |
      v
compute and data modules
  data_catalog | config | spatial | scenario | terrain | visualization | io
      |
      v
file contracts
  data/datasets.yaml | configs/**/*.yaml | .lte-data/cache | run directories
```

Interface modules parse CLI arguments or handle page events. They do not own
scientific logic, path rules, or serialization. Application services expose
typed request, progress, result, and error objects that both interfaces use.

### Package boundaries

The implementation introduces these focused responsibilities:

- `profiles.py`: profile schema, discovery, validation, CRUD, default-profile
  links, and legacy profile conversion;
- `candidate_scanner.py`: lazy deterministic grid traversal, point counting,
  exact geometry checks, spacing, fast and complete selection modes, progress,
  and cancellation;
- `candidate_cache.py`: cache keys, schemas, atomic publication, corruption
  handling, and legacy cache import;
- `selection_service.py`: input preflight, candidate jobs, candidate details,
  DEM statistics, selection locking, and station export;
- `figure_service.py`: CSV or run loading, validated figure specifications,
  preview rendering, and final rendering;
- `run_service.py`: unique run directories, run manifests, artifact records,
  parent-child links, and history discovery;
- `map_assets.py`: boundary display geometry, station display data, cached DEM
  overviews, candidate-window overlays, and offline asset serving;
- `gui/`: NiceGUI application shell, translations, shared components, and page
  adapters only.

Existing modules remain the lower-level implementation where their current
boundaries are sound. The thin files in `scripts/` remain thin.

## Experiment Profile Model

### Schema version 2

New and migrated profiles use a versioned schema that references catalog IDs
instead of repeating registered boundary and DEM paths:

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
  azimuth_deg: -60
  elevation_deg: 30
  vertical_exaggeration: 1.0
  station_color: red
  station_marker_size: 20
  title: null
```

The selected scenario resolves the registered boundary and DEM. The points
dataset is an explicit catalog ID because a future repository can contain more
than one station source. Data acquisition metadata remains only in the catalog.

### Discovery and the default profile

Profiles are discovered recursively below `configs/`. New profiles are written
to `configs/<scenario-id>/<profile-id>.yaml`. Existing files can remain at their
current paths.

The scenario's existing `config_path` field identifies its default profile.
Other profiles identify their owning scenario through `profile.scenario_id`.
This preserves the current catalog shape while supporting multiple profiles.

A ready scenario without a saved profile receives an in-memory draft populated
from explicit application defaults: `EPSG:3857`, a 3000 m rectangle, target 30,
tolerance 0, fast uniform scanning, 10 m step, maximum 100 candidates, minimum
spacing equal to the rectangle size, and seed 42. All current output formats
are enabled. The user can run the draft, in which case the resolved run snapshot
is still written, or save it as a reusable profile.

### Profile editing rules

- The GUI never autosaves a profile.
- An unsaved-change indicator remains visible until an explicit Save or
  Discard.
- Save validates the complete schema before touching the destination.
- Writes use a temporary sibling file followed by an atomic replacement.
- Profile IDs are lowercase, filesystem-safe slugs and are unique within a
  scenario.
- Copy creates a new ID and never changes the source.
- Rename writes the new file before removing the old file.
- Renaming or replacing a default profile updates `config_path` under the same
  repository transaction lock.
- Deleting a default profile is blocked until another profile is selected as
  default.
- Delete and overwrite operations require an explicit confirmation dialog.
- GUI changes are ordinary repository changes and remain visible to Git.

Legacy YAML files remain runnable. Opening one in the GUI shows a migration
preview. The first explicit Save converts it to schema version 2 while
preserving its effective experiment values. The CLI reader supports both schema
versions throughout the first release.

## Candidate Scanning

### Exact constraints

A candidate is valid only when:

- its station count is within
  `target_base_station_count +/- count_tolerance`;
- the complete axis-aligned square is strictly inside the registered boundary,
  preserving the current `contains` semantics;
- its center is at least `minimum_center_spacing_m` from every already selected
  candidate center under the mode's deterministic selection rule.

No DEM statistic participates in validity or ordering.

### Memory-bounded row sweep

The scanner does not allocate an `N x 2` array for all Cartesian grid
positions. It builds only the one-dimensional X and Y origin arrays. For each Y
row or small row chunk it:

1. identifies stations whose Y coordinate lies inside the rectangle height;
2. sorts or incrementally maintains their X coordinates;
3. uses vectorized `searchsorted` boundaries to obtain the exact inclusive
   station count for all X positions in that row;
4. constructs Shapely rectangles only for count-matching positions;
5. applies an exact vectorized boundary-containment check;
6. applies deterministic spacing and mode selection rules;
7. emits progress and provisional-candidate events.

This makes working memory proportional to the station set, one grid axis, the
configured result limit, and the row chunk rather than the complete Cartesian
grid.

Sequential strategy visits axes in coordinate order. Uniform strategy applies
seeded deterministic permutations to the one-dimensional axes. This does not
reproduce the old full-grid permutation, which is why every cache and run
record includes a scanner algorithm version.

### Fast mode

Fast mode emits valid candidates in deterministic traversal order and stops as
soon as `max_rectangles` candidates satisfy all constraints. It is the default
interactive mode.

### Complete mode

Complete mode evaluates every grid position. Each valid position receives a
stable 64-bit priority derived from the seed and flat grid ID. A bounded online
selection set keeps at most `max_rectangles` candidates while enforcing minimum
spacing. A higher-priority candidate can deterministically replace lower-
priority conflicting candidates. Equal priorities are resolved by flat grid ID.
The UI marks complete-mode candidates as provisional until the final position
has been evaluated.

This mode examines the full search space without retaining every valid
rectangle. It is deterministic but is not presented as an optimization of a
scientific quality objective.

### Progress and cancellation

The scanner emits structured events containing:

- phase;
- checked and total grid positions;
- progress fraction;
- current and maximum candidate counts;
- elapsed time;
- cache status;
- provisional candidate additions or replacements.

The worker checks a cancellation token between row chunks. Cancelled jobs do
not publish a formal candidate cache. Their provisional candidates disappear
when the page leaves the cancelled job.

Only one candidate scan or final figure-rendering job runs at a time. The GUI
remains responsive and can show other read-only pages while a job runs.

## Candidate Cache

Candidate caches live below `.lte-data/cache/candidates/` rather than inside a
run directory. Each cache has a schema version and is addressed by a SHA256 key
over:

- scenario ID;
- boundary and station dataset fingerprints from the manifest;
- target CRS;
- rectangle size, target count, and tolerance;
- scan mode, strategy, step, maximum result count, minimum spacing, and seed;
- candidate-scanner algorithm version.

DEM identity is excluded because DEM data does not affect candidate validity.
DEM overviews and statistics use separate keys that include the DEM
fingerprint.

On a hit, the service validates the cache schema, key, candidate count fields,
and geometry reconstruction before returning it. A malformed or inconsistent
cache is quarantined with a diagnostic and regenerated. Cache writes use a
temporary file and atomic replacement. Force Rescan bypasses reads but replaces
the same key only after a successful scan.

Legacy output-directory cache JSON can be imported when its filename and
effective configuration match. Imported candidates are revalidated against the
current station coordinates and boundary before a new versioned cache is
published.

## GUI Information Architecture

The application uses a desktop-browser layout with a persistent top bar for
navigation, active-job state, language, and application status.

### Scenarios

The landing page lists every catalog scenario with display name, readiness,
boundary, DEM, default profile, and the most recent validation result.

- `ready` scenarios enable Configure and Run.
- `dem-pending`, `boundary-ready`, and `invalid` scenarios remain inspectable
  but cannot start a selection run.
- Validate runs the existing fast validation.
- Full Checksum is a separate, explicitly slow action.
- Registration, DEM export, and DEM ingest commands are shown as copyable CLI
  guidance rather than GUI actions.

Starting a run always performs fast data validation. A derived `ready` status is
therefore necessary but not sufficient when the catalog, manifest, or files
have drifted.

### Configure

The Configure page selects a scenario profile, supports profile CRUD, edits
grouped parameters, chooses the output root, and performs preflight validation.
Basic parameters are visible by default. Tolerance, scan strategy, seed,
minimum spacing, and complete mode are under an Advanced section.

The output-root field supports validated manual entry. A Browse button opens a
host operating-system directory dialog when the Python environment provides
one; manual entry remains the cross-platform fallback.

Starting a run freezes an immutable in-memory configuration snapshot. Later
profile edits cannot change an active job.

### Candidate Explorer

The default view dedicates most space to the Leaflet map, with layer controls
on the left and one selected-candidate detail panel on the right. A toolbar
switch adds or removes the bottom candidate filmstrip. Both layouts share map
bounds, layer state, active candidate, and scan progress.

Map layers include:

- cached local DEM overview, defaulting to elevation color plus hillshade;
- elevation-only and hillshade-only DEM modes;
- DEM opacity;
- registered boundary;
- stations;
- candidate rectangles;
- an optional online basemap that is disabled cleanly when offline.

Candidate rectangles are red by default and green when selected. Hover shows
candidate ID and station count. Click selects exactly one candidate. Next,
Previous, and direct candidate-ID navigation resolve dense or overlapping
rectangles without requiring pixel-perfect clicks.

The detail panel shows count, center, bounds, elevation statistics, and scan
parameters. DEM statistics are computed lazily by block-window streaming over
the registered raster and cached per candidate. Station hover or click shows
cell ID, longitude, latitude, range, samples, timestamps, and selected-candidate
elevation when available.

Scan candidates appear progressively. The Confirm Candidate button stays
disabled until scanning completes and one candidate is selected.

### Generate

Confirm Candidate locks a copy of the candidate and opens the Generate page.
The page summarizes the scenario, profile, candidate, destination, and expected
artifacts. The user independently enables:

- scenario CSV;
- two-dimensional selection preview PNG;
- three-dimensional PNG;
- EPS;
- offline interactive HTML.

Clicking Generate is the first action that creates the final run directory.

### Figures

The Figures page accepts the current locked candidate, a previous run
directory, or a compatible CSV. It provides Preview and Publication presets and
validated controls for colormap, DPI, camera azimuth and elevation, vertical
exaggeration, station marker, and title.

Refresh Preview renders a bounded low-resolution result. Parameter changes do
not trigger rendering automatically. Final Export renders selected formats at
their configured resolution.

### History

History discovers run manifests under the repository results root and every
external output root previously used by this local GUI. A rebuildable local
index accelerates discovery but is never authoritative.

The output-root list, selected language, and other workstation preferences live
in ignored `.lte-data/gui-settings.json`. They are local UI state rather than
experiment configuration.

Each entry shows scenario, profile, timestamps, status, parent run, parameters,
selected candidate, and artifacts. Actions can reveal the directory, inspect
records, reopen figures, or create a derived figure run. The first release does
not delete runs.

## DEM and Map Assets

The browser never receives the full 1 m GeoTIFF. `MapAssetService` produces two
types of display-only assets:

- a cached, georeferenced city overview bounded to a configured maximum pixel
  dimension;
- a cached higher-resolution candidate-window overlay generated on demand at a
  bounded display resolution.

Elevation colors and hillshade are derived from masked raster reads. Transparent
NoData remains transparent. Display asset keys include DEM fingerprint, bounds,
output dimensions, color limits, hillshade parameters, and style version.

Boundary GeoJSON can be simplified for browser display, but all scientific
containment tests use the unsimplified registered geometry. Station data sent to
the browser contains only required display attributes and the current city
subset.

NiceGUI, Leaflet, Plotly, fonts, and application assets are served locally.
Interactive HTML output embeds or references a copied local Plotly asset and
does not use a CDN.

## Output and Run Model

### Unique directories

The user selects an output root. Every generation creates:

```text
<root>/<scenario-id>/<profile-id>/<YYYYMMDD-HHMMSS>-<short-run-id>/
```

The random run suffix prevents same-second collisions. Run timestamps are
stored in UTC and displayed in the user's local timezone. Existing directories
are never reused automatically.

### Run contents

A selection run contains:

- `run-config.yaml`: immutable resolved schema-version-2 snapshot;
- `selection.json`: candidate geometry, statistics, algorithm version, and
  selected station identifiers;
- the current scenario CSV name pattern and selected figure artifacts;
- `run.json`: status, provenance, inputs, outputs, software versions, Git
  commit, cache keys, command or GUI entrypoint, and errors;
- compatibility operation records when an existing CLI consumer expects
  `run-select-sites.json` or `run-generate-figures.json`.

The CSV retains existing columns and adds `run_id`, `scenario_id`, `profile_id`,
and `candidate_id`. CRS and full provenance remain in the run manifest and
configuration snapshot.

### Publication and partial failures

Artifacts are written to temporary sibling paths and renamed only after each
file validates. The run manifest begins in a staging directory. On complete
success the staging directory is atomically renamed to the final run directory.

If the core CSV succeeds but an optional figure fails, the run can be published
with `status: partial`, the successful artifacts, and a structured error for
the missing artifact. The History and Figures pages expose Retry Missing
Artifacts. A run with no valid requested output is not published.

A figure regeneration from History creates a new run with `parent_run_id`; it
does not overwrite the source run.

## Figure Service Improvements

`FigureService` separates four operations:

1. load and validate scenario data;
2. resolve the rectangle and CRS;
3. prepare reusable terrain arrays at a requested preview or final resolution;
4. render requested formats from a validated `FigureSpec`.

The run snapshot is the authoritative CRS source. Legacy CSV without a run
snapshot falls back to `EPSG:3857` with an explicit warning. Required numeric
columns are checked for finite values. A new run contains exactly one rectangle.
If a legacy CSV contains multiple `rect_id` values, the GUI requires the user
to select one instead of silently using the first row.

Preview and Publication presets provide stable defaults. User controls are
validated and recorded in `run.json`. Publication output uses the requested
DPI and optional EPS. Interactive HTML remains self-contained for offline use.

## Error Handling and Job Control

Application services raise typed domain errors with stable codes, a concise
message, and optional details. CLI adapters map them to stderr and exit codes.
GUI adapters map them to field errors, notifications, and expandable technical
details. Services do not print warnings directly.

Preflight verifies profile validity, scenario readiness, registered paths, DEM
readability, output-root writability, free-space availability for known large
outputs, and cache compatibility before a final directory is created.

CPU-intensive scans, DEM statistics, and final figures run outside the NiceGUI
event loop. Workers communicate only through immutable request objects,
progress events, cancellation tokens, and result objects. A process-local job
coordinator enforces one active CPU-intensive job and reports why a second job
cannot start.

Cancelled and failed jobs clean temporary files they own. They never delete
user-selected source data, downloaded DEM shards, profiles, or completed runs.

## Local Security and Offline Behavior

`lte-gui` binds to `127.0.0.1` by default and opens a browser. No authentication
is required for loopback-only use. Binding to a non-loopback address requires
an explicit `--host` value and prints a warning that the application can read
and write local experiment paths.

The GUI does not expose arbitrary file serving. Map and artifact routes resolve
paths against allowlisted cache and run roots and reject traversal. Profile IDs
and output path components are sanitized independently of display names.

No selection, validation, history, local DEM, or figure operation performs a
network request. The optional online basemap is visually marked and can be
disabled. Network failure removes only that layer.

## Internationalization

UI text uses stable translation keys with complete English and Chinese
dictionaries. English is the default. The selected language is stored in local
GUI settings.

Dataset IDs, scenario IDs, profile IDs, configuration keys, paths, CSV columns,
JSON fields, log messages recorded for machines, and exception codes are never
translated. User-facing error summaries are translated while technical details
retain the original English message.

## CLI Compatibility

### `lte-select-sites`

- `--config`, `--city`, `--output-dir`, `--size`, `--target`, and
  `--select-index` remain accepted.
- For a schema-version-2 profile, `--city` must resolve to the profile's
  `scenario_id`; cross-city profile overrides are rejected. Legacy profiles
  retain their current city override behavior.
- Existing `--output-dir` remains an exact final-directory override for legacy
  automation, but conflicting artifact names fail instead of being silently
  overwritten. A new `--output-root` option creates the unique
  scenario/profile/run hierarchy used by the GUI.
- Without `--select-index`, the default selector is the new local web candidate
  explorer.
- `--selector legacy` opens the existing Matplotlib selector.
- `--selector web` explicitly requests the web selector.
- Headless environments receive an actionable error that recommends
  `--select-index` or `--selector legacy` only when a supported display exists.
- Legacy and schema-version-2 profiles resolve through the same service.

### `lte-generate-figures`

- Existing config-based invocation remains accepted.
- New options can load a run directory, choose a rectangle from a legacy
  multi-rectangle CSV, select a preset, and override supported figure fields.
- It calls `FigureService` and writes the same run and artifact schemas as the
  GUI.

### `lte-gui`

The new entrypoint starts the loopback server, opens the default browser unless
disabled, accepts a repository/catalog path and port, and reports how to stop
the process. GUI dependencies are installed through a `gui` optional extra. If
the entrypoint is invoked without that extra, it prints the exact installation
command instead of an import traceback.

## Migration

Migration is incremental and reversible at the artifact boundary:

- existing catalog and dataset paths remain authoritative;
- `config_path` retains its meaning as the default profile;
- legacy YAML remains readable and changes only after an explicit GUI Save;
- current output filenames remain usable inside the new unique run directory;
- old candidate cache JSON can be revalidated and imported;
- legacy run records and CSV files remain discoverable;
- the legacy Matplotlib selector remains available;
- no large DEM or vector dataset is moved;
- `.superpowers/` is added to `.gitignore` so visual design sessions remain
  local, matching the existing treatment of `docs/superpowers/`.

## Testing Strategy

### Unit tests

- schema-version-2 parsing, validation, defaults, and serialization;
- legacy effective-value equivalence and explicit migration;
- profile create, copy, rename, default update, overwrite protection, delete
  protection, rollback, and concurrent-change detection;
- row-sweep counts against a brute-force oracle on small grids, including
  points on rectangle edges;
- exact boundary containment and minimum spacing;
- fast and complete modes, deterministic seeds, progress, replacement, and
  cancellation;
- proof that the scanner does not materialize the complete Cartesian grid;
- cache-key sensitivity, hits, forced refresh, corruption quarantine, atomic
  writes, and legacy import;
- masked DEM overview, hillshade, candidate statistics, and cache invalidation;
- single-rectangle CSV validation, CRS resolution, legacy multi-rectangle
  selection, figure presets, and preview/final separation;
- run-directory uniqueness, staging, partial status, parent links, and history
  discovery;
- path containment, slug validation, and non-loopback warnings;
- complete English and Chinese translation-key parity.

### Integration tests

- ready scenario -> profile snapshot -> candidate scan -> manual index selection
  -> CSV and run publication;
- cache reuse and forced rescan through both CLI and GUI adapters;
- candidate -> low-resolution preview -> final PNG/HTML publication;
- existing run -> derived figure run -> history parent link;
- fast and full scenario validation surfaced through the GUI adapter;
- legacy CLI arguments, YAML, cache, CSV, and run-record fixtures;
- NiceGUI route and component smoke tests with all external network access
  disabled.

### Repository verification

The implementation must keep the existing verification bundle green and add
GUI-specific tests without requiring the full local DEMs in CI:

```powershell
python -m ruff check src scripts tests
python -m pytest -q
python -m compileall -q src/lte_scenario_toolkit scripts
git diff --check
```

Tests use small vector and raster fixtures. An opt-in local benchmark exercises
the real Chicago and New York grid dimensions without entering CI.

## Acceptance Criteria

The design is complete when an implementation satisfies all of the following:

1. `lte-gui` starts a loopback-only, bilingual local web application without
   Node.js or an internet connection.
2. Every catalog scenario is visible with status; only ready scenarios can run.
3. Users can safely manage multiple repository YAML profiles and one default per
   scenario.
4. Identical candidate inputs and algorithm versions yield identical results
   across CLI and GUI.
5. Scanning never allocates the complete Cartesian position array, reports
   progress, supports cancellation, and publishes only complete caches.
6. Users can inspect candidates over local DEM terrain, switch between the map
   and filmstrip layouts, and select exactly one candidate without ranking.
7. Confirmation and generation are separate actions, and every generation uses
   a unique run directory.
8. CSV, two-dimensional preview, static three-dimensional, EPS, and offline HTML
   outputs are individually selectable and traceable.
9. Existing runs or compatible CSV files can produce preview and publication
   figures without rescanning candidates.
10. Run history requires no database and can rebuild itself entirely from run
    files.
11. Existing CLI use, legacy YAML, legacy caches, CSV files, run records, and the
    Matplotlib selector have documented compatibility paths.
12. No GUI action registers a scenario, submits an Earth Engine export, or
    ingests DEM shards.
13. Unit, integration, CLI compatibility, offline, and repository verification
    tests pass.

## Implementation Order

Implementation proceeds through vertical, independently testable slices:

1. versioned profiles and shared resolved-run models;
2. row-sweep scanner, progress, cancellation, and candidate cache;
3. selection, DEM statistics, run publication, and figure service;
4. NiceGUI shell, scenario status, validation, and profile management;
5. DEM map assets and candidate explorer;
6. generation, figures, and run history pages;
7. default web CLI selector, legacy compatibility, documentation, and final
   verification.

No GUI page is allowed to introduce a second implementation of a service that
already exists in the shared package layer.
