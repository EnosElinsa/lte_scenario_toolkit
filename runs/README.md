# Run artifacts

The repository tracks only reusable run templates and concise summaries. Complete generated `code.py` files, temporary logs, caches, and large figures remain local.

- `templates/`: reusable run-record templates.
- `summaries/`: short summaries of completed experiments.
- Individual run directories: local by default and excluded from Git.

After scenario selection or figure generation succeeds, the program writes `run-select-sites.json` or `run-generate-figures.json` to the result directory. These JSON files are machine-readable records of individual runs. To publish an experiment, copy its essential metadata into `summaries/` instead of committing large results or complete caches.
