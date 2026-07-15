# Data contract

## Tracked inputs

### `boundary_shp/`

当前边界数据总量约 0.8 MiB，仓库所有者已确认具有公开再分发权限，因此直接纳入 Git。每个 Shapefile 目录中的 `.shp`、`.shx`、`.dbf`、`.prj`、`.cpg` 以及可用的索引/元数据文件应保持完整。

当前工作流统一使用 `EPSG:3857`，即 WGS 1984 Web Mercator Auxiliary Sphere。使用其他边界时，运行前需要检查 CRS、几何类型、要素数量和边界名称。

### `points_shp/`

完整 LTE 基站点数据约 77 MiB，仓库所有者已确认可以公开。大型二进制文件通过 Git LFS 管理，克隆后执行：

```powershell
git lfs pull
```

Shapefile 组成文件必须一起保留。数据字段、来源和授权以目录内元数据及项目发布说明为准。

## External inputs

### `dem/`

完整 DEM 不进入 Git，也不使用 Git LFS。Chicago DEM 和纽约市 1 m DEM 都应通过公开来源或仓库内下载脚本准备。下载后的文件只存在于本地 `dem/`，该目录已被 `.gitignore` 忽略。

DEM 下载和投影约定见 [dem/README.md](../dem/README.md)。

## Dataset terms

MIT License 只适用于本仓库源代码和文档。数据集仍受其来源、元数据和再分发授权约束；使用者应在论文、报告和二次发布中保留数据来源与版权说明。

## Reproducibility manifest

后续正式数据清单应记录以下字段：

```text
dataset_id
source_url
provider
license
download_date
file_size_bytes
sha256
crs
spatial_resolution
notes
```
