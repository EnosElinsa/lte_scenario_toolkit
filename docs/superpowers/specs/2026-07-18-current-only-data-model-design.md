# Current-Only Data Model Design

Date: 2026-07-18
Status: Approved in conversation

## Goal

Remove unused compatibility concepts from the toolkit so every user-facing and
persisted workflow has one current, unversioned representation.

## Product Boundary

The toolkit has no released schema-version-1 user base to preserve. Existing
local caches, GUI settings, history indexes, and old run records may be ignored
and rebuilt. The implementation therefore makes a clean break instead of
shipping migration code, hidden compatibility readers, or a conversion command.

Repository-provided profiles remain available after being rewritten into the
single current profile shape. User-created profile files remain local and
untracked.

## Canonical Persisted Data

Remove the `schema_version` field from every persisted document produced or
consumed by the package, including:

- experiment profiles;
- the data catalog and scenario manifest;
- candidate cache payloads and selection sidecars;
- GUI settings and history indexes;
- run records and run configuration snapshots;
- figure-generation records.

Readers validate required fields, field types, allowed values, path safety, and
cross-resource identity directly. They do not accept multiple document shapes
or select a parser through a numeric version discriminator. Existing local
documents that no longer match the current structure are skipped or rebuilt at
the normal cache/index/settings boundary; they are not rewritten in place.

## Profiles and Configure Page

`ExperimentProfile` represents only the current nested profile structure. Remove
legacy flat-YAML discovery, effective-value previews, source revision hashing,
explicit conversion, concurrent migration checks, and migration-only GUI states.

The Configure page either loads a valid current profile or creates a normal
in-memory draft from catalog defaults. Invalid profile files produce the usual
validation error and do not expose a migration workflow. Saving always writes
the same unversioned current structure.

Rewrite repository example profiles and their documentation into this structure.
Do not add a compatibility alias for the old layout.

## Figures Sources

The Figures page accepts exactly two source types:

1. the current confirmed selection; or
2. a completed toolkit run directory or its `run.json`.

A run may internally reference CSV and DEM artifacts, but a bare CSV is not a
user-selectable source. Remove standalone CSV inspection, multi-rectangle CSV
selection, DEM attachment, and the `Attach DEM` field and action. A source that
does not carry recorded DEM provenance is rejected as unsupported.

## CLI and Runtime Compatibility

The selection CLI keeps the web selector and deterministic `--select-index`
headless selection. Remove the Tk/Matplotlib selector, `--selector legacy`,
legacy candidate-result translation, legacy cache import, exact legacy output
directory relocation, compatibility output labels, and their guidance text.

Run discovery consumes only current `run.json` records. Remove normalization of
older operation records and history actions that infer artifacts from legacy
metadata. Old run directories are ignored rather than surfaced as partially
functional history entries.

Compatibility helpers that exist solely to reproduce old Cartesian result
shapes or filenames are removed. Current scan algorithms, generated artifact
content, and the web candidate page remain unchanged.

## Failure and Rebuild Behavior

- A stale candidate cache entry is a cache miss and is recomputed.
- Invalid GUI settings fall back to defaults and are rewritten on the next save.
- An invalid history index is rebuilt by scanning current run records.
- Old or invalid run records are omitted from History.
- Invalid current profile, catalog, or manifest input reports a structural
  validation error because these are user-maintained authoritative inputs.
- No loader mutates an old document as a side effect of reading it.

## Documentation and Tests

Delete legacy fixtures and tests whose only purpose is compatibility. Replace
schema-version assertions with structural validation assertions, and add
regressions proving newly written documents omit `schema_version`, Figures
rejects bare CSV sources, CLI help exposes no legacy selector, and old caches or
indexes follow the documented rebuild behavior.

Update README files and examples so users see only the current workflows. Older
historical design documents remain historical records; the new design supersedes
their compatibility requirements.

## Verification

Run focused tests after each subsystem change, followed by Ruff, Python bytecode
compilation, the complete Pytest suite, and `git diff --check`. Restart the local
GUI and verify that Configure, Figures, selection, and History render without
migration, schema-version, bare-CSV, DEM-attachment, or legacy-selector controls.
