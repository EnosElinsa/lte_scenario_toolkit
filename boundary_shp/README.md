# Boundary data

`boundary_shp/` contains the administrative or study-area boundaries used by
registered LTE scenarios. The catalog in `data/datasets.yaml` is the source of
truth for each boundary path, entrypoint, CRS, feature count, provider, and
redistribution terms.

## Add a boundary

`lte-data scenario add` accepts a local path or an HTTP(S) URL for a Shapefile,
ZIP archive, GeoJSON file, or GeoPackage. Use `--layer` when an archive or
GeoPackage contains more than one vector layer. Always provide the provider and
license text. Use `--redistribution-confirmed` only when the repository may
redistribute the normalized boundary.

```powershell
lte-data scenario add boston `
  --boundary-source D:\data\boston-boundary.zip `
  --provider "U.S. Census Bureau" `
  --license "Public domain; source attribution retained" `
  --redistribution-confirmed
```

The command stages the source, records a source checksum, validates the layer,
and installs it atomically. A source must declare a CRS, contain at least one
feature, and contain only nonempty valid `Polygon` or `MultiPolygon` geometries.
The selected features are dissolved into one study boundary and normalized to
the repository's metre-based `EPSG:3857` workflow. The catalog remains the
authority for the declared CRS of every existing entry.

For Shapefiles, keep the `.shp`, `.shx`, `.dbf`, `.prj`, and `.cpg` components
together; optional index and metadata files may remain beside them. The
registration transaction updates the catalog and manifest only after the
boundary directory is complete. A failed import leaves the previous catalog
and installed data untouched.

## Registered scenarios

The current catalog contains these scenario IDs:

- `phoenix`
- `chicago`
- `chicago-cbd`
- `cambridge`
- `new-york-city`

Registration creates a boundary dataset, a pending external DEM declaration,
and a scenario link. If the source URL or acquisition date is unknown, the
catalog stores `null`; values are never inferred from a filename or a search
result. Inspect the resulting links with:

```powershell
lte-data scenario list
lte-data scenario show chicago
```
