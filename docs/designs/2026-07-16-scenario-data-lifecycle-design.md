# Scenario Data Lifecycle Design

Date: 2026-07-16
Status: Approved in conversation

## Goal

Provide one reproducible, configuration-aware workflow for adding an LTE study
area, exporting its USGS 3DEP 1 m DEM from Google Earth Engine, ingesting the
manually downloaded Drive shards, registering data provenance, and validating
the finished scenario.

Chicago, New York City, and future study areas must use the same implementation.
The registered local boundary is the single source of truth for both scenario
selection and the Earth Engine export region.

## Current Problem

The repository currently documents only the Chicago and New York City DEMs.
The New York exporter embeds city-specific TIGER county logic, while scenario
selection reads a different local Shapefile. This creates a risk that the DEM
and the scenario boundary do not cover exactly the same geometry.

`boundary_shp/` contains several study areas but has no documented or validated
extension procedure. `data/datasets.yaml` groups every boundary into one dataset,
so it cannot preserve source, license, checksum, and readiness information for
each study area.

## Design Principles

- Use the registered local boundary as the only authoritative ROI.
- Keep data acquisition metadata separate from experiment scanning parameters.
- Never start an Earth Engine export without an explicit `--export` flag.
- Never hard-code a Cloud Project ID, city name, county code, or private path.
- Validate data before changing the registry or final destination files.
- Make registry and destination updates atomic.
- Keep large DEMs outside Git while retaining provenance and checksums.
- Require an explicit redistribution confirmation before writing a boundary to
  the Git-tracked `boundary_shp/` directory.
- Preserve the current Chicago and New York physical file paths to avoid moving
  multi-gigabyte local rasters. New scenarios use deterministic lowercase IDs.

## Architecture

The existing experiment YAML files remain responsible for rectangle size,
station target count, scan strategy, output options, and other experiment
parameters. A new data lifecycle CLI owns acquisition and validation:

```text
local vector or official URL
        |
        v
lte-data scenario add
        |
        v
registered normalized boundary
        |
        v
lte-data dem export --dry-run / preflight / --export
        |
        v
manual Google Drive shard download
        |
        v
lte-data dem ingest
        |
        v
disk-backed merged DEM and registry update
        |
        v
lte-data validate
```

The implementation is divided into focused package modules:

- `data_catalog.py`: schema loading, cross-reference checks, stable writes, and
  manifest updates.
- `boundary_data.py`: source acquisition, archive handling, vector validation,
  normalization, and Earth Engine ROI conversion.
- `dem_data.py`: Earth Engine preflight/export, shard validation, disk-backed
  merging, and raster coverage checks.
- `data_cli.py`: argument parsing, user-facing status, and exit codes.

The package exposes one new console command:

```toml
lte-data = "lte_scenario_toolkit.data_cli:main"
```

The following city-specific interfaces are removed without compatibility
aliases:

- `lte-download-newyork-dem`;
- `scripts/download_newyork_1m_dem.py`;
- `src/lte_scenario_toolkit/newyork_dem.py`;
- `gee/newyork_1m_dem.js`.

## Registry Schema

`data/datasets.yaml` becomes schema version 2 and remains the canonical editable
registry. `data/manifest.json` remains generated output.

Each boundary is registered as an independent dataset with:

- a stable dataset ID;
- `role: boundary`;
- the component directory and exact Shapefile entrypoint;
- source URL or source-file checksum;
- provider, license, download date, and redistribution statement;
- stored CRS, geometry type, feature count, and notes.

Each DEM is registered independently with:

- a stable dataset ID;
- `role: dem` and `external: true`;
- its directory and exact GeoTIFF entrypoint;
- Earth Engine collection, band, units, vertical datum, native scale, export
  CRS, and expected local path;
- the latest successful Earth Engine task ID and export timestamp when known.

The top-level `scenarios` list links the two dataset roles and an optional
experiment configuration:

```yaml
schema_version: 2

datasets:
  - dataset_id: boundary_chicago
    role: boundary
    path: boundary_shp/Chicago
    entrypoint: boundary_shp/Chicago/Chicago_Boundary.shp
    # provenance and spatial metadata

  - dataset_id: usgs_3dep_1m_dem_chicago
    role: dem
    path: dem/USGS_1M_DEM_Chicago
    entrypoint: dem/USGS_1M_DEM_Chicago/USGS_1M_DEM_Chicago.tif
    external: true
    # Earth Engine and raster metadata

scenarios:
  - scenario_id: chicago
    display_name: Chicago
    boundary_dataset_id: boundary_chicago
    dem_dataset_id: usgs_3dep_1m_dem_chicago
    config_path: configs/example.yaml
```

Readiness is derived rather than stored. A scenario can be `boundary-ready`,
`dem-pending`, or `ready` according to the registered files and validation
results. Phoenix, Cambridge, and Chicago CBD can therefore be represented
accurately before a DEM or experiment configuration exists.

### Existing-data migration

The aggregate `administrative_boundaries` record is replaced by one boundary
dataset per existing directory. Five scenario mappings are created for Chicago,
Chicago CBD, New York City, Phoenix, and Cambridge. Chicago and New York City
link to their existing DEM datasets and experiment configurations. The other
three scenarios receive pending DEM declarations and no default experiment
configuration. Existing vector and raster files remain at their current paths;
the migration changes metadata and references, not multi-gigabyte data.

## Command Design

### Register a scenario and boundary

```powershell
lte-data scenario add chicago `
  --boundary-source <local-path-or-official-url> `
  --provider "<provider>" `
  --license "<license>" `
  --redistribution-confirmed
```

Scenario IDs must match `[a-z][a-z0-9-]*`. The command accepts a local or HTTP(S)
Shapefile, ZIP archive, GeoJSON, or GeoPackage. A source containing more than one
usable vector layer requires `--layer`.

The command performs all work in a staging directory:

1. Download or copy the source and compute its SHA256.
2. Extract archives with path-traversal and symlink protection.
3. Read the selected layer and require a declared CRS.
4. Require non-empty Polygon or MultiPolygon geometry.
5. Reject invalid geometry with an actionable report; do not silently repair it.
6. Dissolve valid features into one authoritative study-area geometry.
7. Reproject the normalized boundary to EPSG:3857.
8. Write a minimal Shapefile containing the scenario ID, display name, and
   geometry.
9. Validate every required Shapefile sidecar.
10. Atomically install the boundary and update the registry.

For new scenarios, the default paths are:

```text
boundary_shp/<scenario-id>/<scenario-id>.shp
dem/<scenario-id>/usgs_3dep_1m_<scenario-id>.tif
```

The DEM registry entry is created in a pending state at the same time. Existing
Chicago and New York paths remain explicitly registered instead of being moved.

`--redistribution-confirmed` records the user's assertion; the program does not
attempt to infer licensing from a URL. A provider and license are mandatory.
Existing destinations are never replaced unless a future design explicitly
adds a separately reviewed replacement workflow.

### Export a DEM from Earth Engine

```powershell
$env:EE_PROJECT = "<cloud-project-id>"

lte-data dem export chicago --dry-run
lte-data dem export chicago
lte-data dem export chicago --export
```

Project resolution order is `--project`, then `EE_PROJECT`, then
`GOOGLE_CLOUD_PROJECT`. No resolved project value is written into source files
or the data registry.

Modes have deliberately different side effects:

- `--dry-run` performs local catalog, boundary, output-path, scale, and estimated
  pixel-count checks without importing or contacting Earth Engine.
- A plain invocation initializes Earth Engine, verifies that
  `USGS/3DEP/1m` intersects the ROI, reports the intersecting image count, and
  prints the resolved plan without starting a task.
- `--export` repeats the online preflight and then submits the Drive task.

The exporter converts the normalized local boundary to EPSG:4326 in memory and
then to an Earth Engine geometry. It filters `USGS/3DEP/1m` by that geometry,
mosaics the intersecting images, selects `elevation`, and clips once at the end.
The default export uses 1 m scale, EPSG:3857, GeoTIFF, cloud-optimized shards,
8192-pixel file dimensions, a 256-pixel shard size, and a `1e13` max-pixel guard.
Empty-tile skipping is explicitly disabled so the expected output grid can be
validated during ingest. The export remains configurable where the existing New
York workflow already exposes a safe parameter.

Every online preflight and export writes a timestamped run directory containing
the resolved plan, dataset semantics, source links, software versions, Git
commit, boundary checksum, and Earth Engine task ID when a task is submitted.

### Ingest downloaded Drive shards

```powershell
lte-data dem ingest chicago --tiles-dir D:\downloads\chicago-dem
```

Only files matching the registered export prefix are considered. The command
requires consistent CRS, pixel size, band count, dtype, NoData value, and grid
alignment. It rejects unrelated files, conflicting overlaps, missing expected
grid cells, and mixed export runs. NoData outside the exact polygon is expected
and is not treated as a missing shard.

The merge is disk-backed and block-oriented. It must not allocate an array for
the complete city raster. Output is first written beside the registered target
as a unique partial file using tiled BigTIFF, LZW compression, and appropriate
floating-point predictor settings. Internal overviews are built after the base
image succeeds.

Before installation, the command verifies:

- the merged raster opens cleanly;
- its CRS and resolution match the registry;
- its extent covers the registered boundary;
- valid elevation samples exist inside the boundary;
- its band and dtype match the declared data layer;
- its SHA256 and file size can be recorded.

Only then is the partial file atomically renamed to the registered entrypoint
and the affected manifest record updated. Downloaded shards are never deleted.

### Validate registered scenarios

```powershell
lte-data validate chicago
lte-data validate chicago --full-checksum
lte-data validate --all
```

Fast validation checks:

- catalog schema, unique IDs, and scenario cross-references;
- repository-relative paths and path containment;
- boundary components, CRS, geometry type, validity, and feature count;
- DEM existence, raster metadata, grid properties, boundary coverage, and valid
  samples;
- linked experiment configuration paths against the registered boundary and DEM;
- manifest presence and recorded file metadata.

`--full-checksum` additionally streams every registered file and compares its
SHA256. It is opt-in because an 11 GiB DEM is expensive to hash. `--all` reports
every scenario in a stable order and returns a non-zero status if any requested
scenario fails.

Read-only `lte-data scenario list` and `lte-data scenario show <id>` commands
display derived readiness and resolved paths without changing state.

## Failure Handling and Atomicity

- Invalid arguments, data-contract failures, and Earth Engine failures are
  reported separately with a suggested next command.
- Archive extraction and downloaded content remain in a staging directory until
  validation succeeds.
- Registry writes use a temporary file, stable YAML serialization, an mtime
  check against concurrent modification, and `Path.replace()`.
- Final vector and raster files use the same staging-and-replace pattern.
- Registry entries are never committed before their corresponding boundary is
  installed successfully.
- DEM export changes only run records and Earth Engine state; it does not claim
  that a local DEM exists.
- DEM ingest does not change the registry or manifest if validation or merge
  fails.
- Existing tracked or external data is never deleted automatically.
- A plain DEM export invocation cannot start a task; only `--export` may call
  `task.start()`.

## Documentation Changes

- Add `boundary_shp/README.md` with accepted source formats, licensing
  requirements, naming rules, and the complete registration example.
- Rewrite `dem/README.md` around the generic export and ingest workflow, using
  Chicago and New York as examples of the same command.
- Update `README.md`, `data/README.md`, `configs/README.md`, `scripts/README.md`,
  and `src/README.md` to describe the new CLI and source-of-truth rule.
- Remove instructions for the New York-specific script and JavaScript exporter.
- Keep all tracked public documentation and source text in English.

## Testing Strategy

Unit and CLI tests use temporary directories and small synthetic vectors and
rasters. CI never contacts Earth Engine or reads the full local DEMs.

Required coverage includes:

- schema version 2 parsing, duplicate IDs, broken cross-references, and stable
  writes;
- local and URL-source staging with mocked HTTP responses;
- safe ZIP extraction and rejection of traversal entries;
- format and layer selection, CRS requirements, geometry validation, dissolve,
  reprojection, and Shapefile component output;
- transactional rollback when boundary installation or catalog writing fails;
- DEM dry-run behavior with no Earth Engine import;
- mocked Earth Engine preflight and proof that `task.start()` is reachable only
  through `--export`;
- shard metadata, alignment, overlap, and missing-grid validation;
- disk-backed merge behavior using small fixtures;
- DEM coverage and valid-elevation checks;
- fast versus full-checksum validation;
- Chicago and New York resolving through the same generic code path;
- absence of the old console command, script, module, and JavaScript file;
- help text and exit status for every public subcommand.

Local acceptance runs include Ruff, Pytest, Python compilation, CLI help and
dry-run smoke tests, repository metadata checks, and a live Earth Engine
preflight using a user-supplied Cloud Project. No export task is submitted as
part of automated acceptance.

## Acceptance Criteria

- A new boundary can be imported from a local vector or official URL and is
  independently represented in the registry.
- Chicago and New York use `lte-data dem export` with no city-specific code.
- The exact registered boundary is used for both site selection and DEM export.
- A Drive shard directory can be merged without loading the complete raster into
  memory.
- Validation clearly distinguishes boundary-ready, DEM-pending, and ready
  scenarios.
- Registry, manifest, boundary, and DEM destinations cannot be left partially
  updated by a handled failure.
- No Cloud Project ID or local absolute path is committed.
- DEM rasters and generated run directories remain ignored by Git.
- All automated tests pass without Earth Engine network access.

## Non-goals

- Automatically infer an administrative boundary from a place name.
- Automatically authenticate to or download files from Google Drive.
- Automatically decide whether a boundary license permits redistribution.
- Upload local boundaries as permanent Earth Engine Assets.
- Support a DEM collection other than USGS 3DEP 1 m in the first version.
- Replace existing scenarios or delete source/downloaded files.
- Change the LTE station-selection or figure-generation algorithms.
