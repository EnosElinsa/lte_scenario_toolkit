# DEM data

完整 DEM 不纳入 Git。这个目录只用于本地存放下载后的栅格文件；`.gitignore` 会忽略其中的大型文件。

## Chicago

现有 Chicago DEM 使用 USGS National Map 3DEP 1 m 数据，当前本地约 10.96 GiB。使用本地场景脚本前，需要把文件放在：

```text
dem/USGS_1M_DEM_Chicago/USGS_1M_DEM_Chicago.tif
```

## New York City

数据集：`USGS/3DEP/1m`

- 波段：`elevation`
- 单位：米
- 垂直基准：NAVD88
- 原始影像：USGS 3DEP 1 m ImageCollection
- 输出 CRS：`EPSG:3857`
- 输出方式：Google Drive 分片 GeoTIFF

使用 Python 导出：

```powershell
python scripts/download_newyork_1m_dem.py `
  --project gen-lang-client-0153149292 `
  --dry-run

python scripts/download_newyork_1m_dem.py `
  --project gen-lang-client-0153149292 `
  --export
```

也可以把 `gee/newyork_1m_dem.js` 粘贴到 GEE Code Editor，在 `Tasks` 面板中启动导出。

默认范围是纽约市五县联合边界：Bronx、Kings、New York、Queens、Richmond。只导出 Manhattan/纽约县时，Python 使用 `--boundary-mode county --county-geoid 36061`，JavaScript 将 `boundaryMode` 改为 `'county'`。

城市级 1 m DEM 通常会生成多个分片。下载完成后，将分片放入本地 `dem/`，必要时用 QGIS 或 Rasterio 合并；不要把合并后的大文件提交到 GitHub。

## Official sources

- [USGS 3DEP 1m Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/USGS_3DEP_1m)
- [Earth Engine Export.image.toDrive](https://developers.google.com/earth-engine/apidocs/export-image-todrive)
