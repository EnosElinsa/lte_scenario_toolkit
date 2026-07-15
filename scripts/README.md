# Command-line wrappers

The supported data lifecycle command is the installed `lte-data` entrypoint.
It manages scenario registration, DEM export planning/submission, shard
ingest, and validation:

```powershell
lte-data scenario add <scenario-id> --boundary-source <path-or-url> --provider <name> --license <terms>
lte-data scenario list
lte-data dem export <scenario-id> --dry-run
lte-data dem ingest <scenario-id> --tiles-dir <download-directory>
lte-data validate <scenario-id>
```

`manage_data.py` is the thin source-tree wrapper for `lte-data`; it adds
`src/` to `sys.path` and delegates to `lte_scenario_toolkit.data_cli.main`.
The other wrappers serve the reproducible research workflow:

- `select_sites.py` delegates to the package scenario selector;
- `generate_scenario_figures.py` delegates to figure generation;
- `create_data_manifest.py` delegates to schema-v2 manifest generation.

Wrappers contain no business logic and should remain small. Add reusable
behavior to the installed package first, then expose it through a wrapper only
when a source-tree invocation is useful. The package console scripts in
`pyproject.toml` are the public interface for installed environments.
