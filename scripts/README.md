# Command-line entry points

- `select_sites.py`: forwards to the configuration-driven scenario selector.
- `generate_scenario_figures.py`: forwards to publication-figure generation for an existing CSV.
- `download_newyork_1m_dem.py`: forwards to the Earth Engine DEM exporter.
- `create_data_manifest.py`: generates file sizes and SHA256 checksums from `data/datasets.yaml`.

These files only add the repository's `src/` directory to the module search path and call the corresponding `main()` function in `lte_scenario_toolkit`. Business logic belongs in the installed package.
