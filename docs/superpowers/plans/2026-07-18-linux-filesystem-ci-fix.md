# Linux Filesystem CI Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the two Ubuntu-only CI failures pass while preserving exact manifest paths and rejecting dangling DEM destination symlinks.

**Architecture:** Keep boundary component matching and manifest validation as separate contracts by making the test manifest reflect the renamed file. Preserve the freshly loaded catalog's resolved containment checks, but add a lexical destination collision check before DEM entrypoint resolution so dangling symlinks cannot disappear through `Path.resolve()`.

**Tech Stack:** Python 3.12, pathlib, pytest, Ruff, GitHub Actions on Ubuntu

---

## File Structure

- `tests/test_data_validation.py`: keep the boundary sidecar fixture's manifest synchronized with its case-only rename.
- `tests/test_dem_data.py`: make the dangling symlink fixture stay inside the declared DEM dataset so it reaches the intended collision contract.
- `src/lte_scenario_toolkit/dem_data.py`: reject an existing lexical DEM entrypoint before resolving symlinks.

### Task 0: Prepare the Linux Feedback Loop

**Files:**
- Verify only; install into the disposable WSL path `/tmp/lte-ci-venv`

- [ ] **Step 1: Create an isolated Linux environment and install the project**

Run:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'rm -rf /tmp/lte-ci-venv && python3 -m venv /tmp/lte-ci-venv && /tmp/lte-ci-venv/bin/python -m pip install --upgrade pip && cd /mnt/c/Users/labs2/Desktop/Projects/lte_scenario_toolkit && /tmp/lte-ci-venv/bin/python -m pip install -e ".[dev]"'
```

Expected: exit code 0 and an editable Linux installation containing pytest and the project dependencies.

### Task 1: Isolate the Case-Insensitive Boundary Sidecar Contract

**Files:**
- Modify: `tests/test_data_validation.py:468-475`

- [ ] **Step 1: Confirm the existing Linux failure**

Run:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'cd /mnt/c/Users/labs2/Desktop/Projects/lte_scenario_toolkit && /tmp/lte-ci-venv/bin/python -m pytest -q tests/test_data_validation.py::test_boundary_sidecar_extension_matching_is_case_insensitive'
```

Expected: FAIL because the generated manifest still records `boundary_shp/city/city.cpg` after the file is renamed to `city.CPG`.

- [ ] **Step 2: Synchronize the test manifest with the actual renamed sidecar**

Replace the test body with:

```python
def test_boundary_sidecar_extension_matching_is_case_insensitive(tmp_path):
    _, catalog = _write_catalog(tmp_path)
    cpg = catalog.resolve("boundary_shp/city/city.cpg")
    uppercase_cpg = cpg.with_suffix(".CPG")
    cpg.rename(uppercase_cpg)
    manifest_path = tmp_path / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    boundary_record = next(
        item for item in manifest["datasets"] if item["dataset_id"] == "boundary_city"
    )
    cpg_record = next(
        item for item in boundary_record["files"] if item["path"].endswith("/city.cpg")
    )
    cpg_record["path"] = uppercase_cpg.relative_to(tmp_path).as_posix()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_scenario_data(catalog, "city")

    assert report.ok
```

- [ ] **Step 3: Run the isolated boundary test**

Run:

```powershell
python -m pytest -q tests/test_data_validation.py::test_boundary_sidecar_extension_matching_is_case_insensitive
```

Expected: PASS on Windows. The later full Linux verification confirms the case-sensitive behavior.

- [ ] **Step 4: Commit the fixture correction**

```powershell
git add tests/test_data_validation.py
git commit -m "test: isolate boundary sidecar case handling"
```

### Task 2: Reject a Dangling DEM Destination Before Resolution

**Files:**
- Modify: `tests/test_dem_data.py:893-921`
- Modify: `src/lte_scenario_toolkit/dem_data.py:1035-1039`

- [ ] **Step 1: Make the regression fixture target the intended collision path**

Change the symlink creation to keep the missing target inside the registered dataset directory:

```python
    try:
        destination.symlink_to(destination.with_name("missing-target.tif"))
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
```

Keep the existing `pytest.raises(DemIngestError, match="already exists")`, lock-call assertion, symlink-preservation assertion, and lock-cleanup assertion.

- [ ] **Step 2: Run the adjusted regression test and verify RED**

Run:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'cd /mnt/c/Users/labs2/Desktop/Projects/lte_scenario_toolkit && /tmp/lte-ci-venv/bin/python -m pytest -q tests/test_dem_data.py::test_ingest_dem_shards_reloads_catalog_only_after_lock_and_rejects_existing_symlink'
```

Expected: FAIL because `Path.resolve()` converts the entrypoint to the missing target before `os.path.lexists()` runs, so the error does not contain `already exists`.

- [ ] **Step 3: Add the minimal lexical collision guard**

In `_ingest_dem_shards_locked`, replace the destination preflight with:

```python
    lexical_destination = catalog.root / Path(dem["entrypoint"])
    if os.path.lexists(lexical_destination):
        raise DemIngestError(f"DEM destination already exists: {lexical_destination}")
    destination = _safe_catalog_path(
        catalog.root, dem["entrypoint"], description="DEM entrypoint"
    )
    if os.path.lexists(destination):
        raise DemIngestError(f"DEM destination already exists: {destination}")
```

The second check remains as defense in depth for the resolved destination and concurrent changes.

- [ ] **Step 4: Run the DEM regression test and verify GREEN**

Run:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'cd /mnt/c/Users/labs2/Desktop/Projects/lte_scenario_toolkit && /tmp/lte-ci-venv/bin/python -m pytest -q tests/test_dem_data.py::test_ingest_dem_shards_reloads_catalog_only_after_lock_and_rejects_existing_symlink'
```

Expected: PASS, with the symlink still present and the transaction lock removed.

- [ ] **Step 5: Run the related DEM ingest tests**

Run:

```powershell
python -m pytest -q tests/test_dem_data.py -k "ingest_dem_shards"
```

Expected: all selected tests PASS; the real-symlink test may SKIP when Windows symlink creation is unavailable.

- [ ] **Step 6: Commit the symlink fix**

```powershell
git add tests/test_dem_data.py src/lte_scenario_toolkit/dem_data.py
git commit -m "fix: reject dangling DEM destinations"
```

### Task 3: Run the GitHub Actions Verification Set

**Files:**
- Verify only; no additional source files expected

- [ ] **Step 1: Run Ruff**

Run:

```powershell
python -m ruff check src scripts tests
```

Expected: exit code 0 with no lint errors.

- [ ] **Step 2: Run entry-point syntax checks**

Run:

```powershell
python -m compileall -q src/lte_scenario_toolkit scripts
```

Expected: exit code 0 with no syntax errors.

- [ ] **Step 3: Run the complete test suite**

Run:

```powershell
python -m pytest -q
```

Expected: exit code 0 with no failed tests.

- [ ] **Step 4: Run the two regressions on Linux**

Run:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'cd /mnt/c/Users/labs2/Desktop/Projects/lte_scenario_toolkit && /tmp/lte-ci-venv/bin/python -m pytest -q tests/test_data_validation.py::test_boundary_sidecar_extension_matching_is_case_insensitive tests/test_dem_data.py::test_ingest_dem_shards_reloads_catalog_only_after_lock_and_rejects_existing_symlink'
```

Expected: `2 passed` on the case-sensitive filesystem environment.

- [ ] **Step 5: Inspect the final diff and repository state**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; only intentional uncommitted plan bookkeeping may remain.
