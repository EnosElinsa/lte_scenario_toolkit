# Tests

The suite exercises the installed package, local GUI, compatibility adapters,
and thin wrappers without live network services or full private/local data.

Coverage includes:

- schema-v2 catalog and profile validation, multiple profiles, defaults,
  guarded CRUD, rollback, and explicit legacy YAML migration;
- memory-bounded deterministic row-sweep scanning, fast/complete modes,
  progress, cancellation, cache reuse, force refresh, and corruption handling;
- boundary registration, mocked Earth Engine preflight/submission, disk-backed
  DEM shard ingest, and fast/full scenario validation;
- one-candidate selection, DEM overlays and statistics, selectable artifact
  generation, partial runs, figure preview/publication, and file-based history;
- legacy YAML, candidate-cache JSON, multi-rectangle CSV, selection/figure run
  records, CLI flags, and the Matplotlib selector;
- NiceGUI routes, bilingual strings, allowlisted files, loopback warnings,
  path containment, and socket-blocked offline flows;
- the opt-in benchmark API with production inputs monkeypatched to tiny
  fixtures, proving that it writes no outputs.

Vector fixtures are small public GeoJSON or temporary Shapefiles. Raster tests
create tiny temporary GeoTIFFs with Rasterio. Earth Engine calls are mocked; CI
does not authenticate, submit exports, download Drive files, read full city
DEMs, or make GUI-core network requests.

Install test and GUI extras, then run the release gates:

```powershell
python -m pip install -e ".[dev,gui]"
python -m ruff check src scripts tests
python -m compileall -q src/lte_scenario_toolkit scripts
python -m pytest -q
```

Focused GUI and legacy compatibility runs:

```powershell
python -m pytest tests/test_gui.py -q
python -m pytest tests/test_profiles.py tests/test_candidate_cache.py `
  tests/test_run_service.py tests/test_figure_service.py -q
```

End-to-end fixture smoke tests publish only below Pytest temporary directories:

```powershell
python -m pytest tests/test_selection_service.py::test_end_to_end_fixture_run -v
python -m pytest tests/test_gui.py::test_offline_candidate_to_history_flow -v
```

Real Chicago and New York scanner benchmarks are intentionally opt-in and are
not part of CI because results depend on local datasets and hardware.
