# Linux Filesystem CI Fix Design

Date: 2026-07-18
Status: Approved in conversation

## Goal

Make the two failing Ubuntu CI tests exercise their intended contracts without
weakening manifest integrity or symlink protections.

## Confirmed Causes

The boundary-sidecar test renames `city.cpg` to `city.CPG` after its fixture has
already generated `data/manifest.json`. On a case-sensitive filesystem, the
manifest then correctly reports its recorded lowercase path as missing, even
though the boundary component scan itself accepts the uppercase extension.

The DEM-ingest test creates a dangling destination symlink whose target is
outside the declared DEM dataset directory. Reloading the catalog under the
transaction lock follows that link and rejects dataset containment before the
ingest collision guard can report that the destination already exists.

## Design

Keep manifest file paths exact. After the boundary test performs its case-only
rename, update that fixture's manifest file record to the actual `.CPG` path.
The test will then isolate the intended case-insensitive Shapefile component
behavior while preserving stale-manifest detection.

Change the DEM regression fixture so its dangling symlink points to a missing
file inside the declared dataset directory. This avoids an unrelated catalog
containment failure and directly exercises the existing-destination contract.

Before resolving the DEM entrypoint through symlinks, derive its lexical path
from the already validated, freshly loaded catalog and reject it with
`os.path.lexists()`. Keep the existing resolved-path containment checks and the
atomic no-clobber installation guard as later defenses.

## Scope

Modify only:

- the two affected regression tests;
- the DEM ingest destination preflight needed to reject a dangling lexical
  destination.

Do not make manifest paths case-insensitive, relax catalog containment, change
the public CLI, or alter unrelated data lifecycle behavior.

## Verification

Use the existing tests as the regression boundary:

1. Run the adjusted DEM symlink test before the implementation change and
   confirm that it fails because the lexical destination is not detected.
2. Run both affected tests after the implementation change.
3. Run Ruff, Python compilation, and the complete Pytest suite using the same
   commands as GitHub Actions.
