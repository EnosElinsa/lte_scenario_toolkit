# Boundary data

Each registered scenario has one authoritative polygon boundary. The same exact
registered entrypoint is used for LTE site selection and as the Earth Engine
(GEE) DEM export region of interest. The catalog in `data/datasets.yaml` is the
source of truth for its path, CRS, feature count, provider, and redistribution
terms.

## Supported sources

`lte-data scenario add` accepts a local path or an HTTP(S) URL for a Shapefile,
ZIP archive, GeoJSON file, or GeoPackage. If a source contains multiple vector
layers, `--layer` is required to select one unambiguously.

## Licensing

The provider, license, and `--redistribution-confirmed` flag are mandatory
because `boundary_shp/` is tracked by Git. Supply the source's real terms and
confirm redistribution only when those terms permit the normalized boundary to
be published in this repository.

## Register a scenario

```powershell
lte-data scenario add boston `
  --boundary-source D:\data\boston-boundary.zip `
  --provider "U.S. Census Bureau" `
  --license "Public domain; source attribution retained" `
  --redistribution-confirmed
```

Registration creates a boundary dataset, a pending external DEM declaration,
and a scenario link. Inspect the resulting links with:

```powershell
lte-data scenario list
lte-data scenario show boston
```

## Validation performed

The command stages the source and verifies that it declares a CRS, contains at
least one feature, and contains only nonempty valid `Polygon` or `MultiPolygon`
geometries. It dissolves the selected features into one boundary and normalizes
the result to `EPSG:3857`.

For Shapefiles, registration installs the `.shp`, `.shx`, `.dbf`, `.prj`, and
`.cpg` sidecars together. It records the source SHA256 and publishes the
boundary, catalog, and manifest changes atomically. A failed import leaves the
previous catalog and installed data untouched.

## Existing data

The current catalog contains these registered scenario IDs:

- `phoenix`
- `chicago`
- `chicago-cbd`
- `cambridge`
- `new-york-city`

If a source URL or acquisition date is unavailable, its catalog field remains
`null`; these values are never inferred from filenames, searches, or local
timestamps.
