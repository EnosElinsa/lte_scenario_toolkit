# Tests

The baseline smoke tests use only repository metadata and small configuration
files. They do not call Earth Engine or require the full DEM.

```powershell
pytest -q
```

Future fixture-based tests will cover CRS conversion, boundary filtering, DEM
sampling, rectangle constraints, CSV fields and error handling for missing
inputs.
