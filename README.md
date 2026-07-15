# USA LTE Base Station Data

面向研究者的可复现实验工具，用于在美国城市行政边界内处理 LTE 基站点、筛选候选空间场景、提取 DEM 高程，并生成 CSV 与二维/三维地形图。

当前仓库处于迁移式重构的第一阶段：现有入口脚本保持可运行，仓库先补齐版本管理、数据说明和可复现运行约定。后续再把重复逻辑拆分到 `src/`，并让配置文件成为正式运行入口。

## Workflow

```text
准备公开的基站点和边界数据
→ 准备本地 DEM 或通过 GEE/USGS 下载
→ 统一到 EPSG:3857
→ 扫描候选矩形
→ 交互选择场景
→ 提取基站点 DEM 高程
→ 输出 CSV 和三维地形图
```

## Repository layout

```text
boundary_shp/                 # 已授权公开的城市边界 Shapefile
points_shp/                   # 公开基站点 Shapefile，Git LFS 管理
dem/                          # 本地 DEM；大文件不进入 Git
configs/                      # 可复现实验配置示例
data/                         # 数据来源和示例数据说明
docs/                         # 设计、计划和研究文档
runs/                         # 只跟踪模板与摘要
select_sites.py               # 候选矩形扫描、交互选择和高程提取
generate_scenario_figures.py  # 从已有 CSV 生成三维图
download_newyork_1m_dem.py    # Python/GEE DEM 导出入口
gee_newyork_1m_dem.js         # GEE Code Editor DEM 导出脚本
```

## Installation

要求 Python 3.10 或更高版本。Windows PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

如果只是运行已有脚本，也可以直接安装运行依赖：

```powershell
python -m pip install geopandas shapely numpy pandas rasterio matplotlib plotly
```

基站点数据使用 Git LFS：

```powershell
git lfs install
git lfs pull
```

## Data preparation

完整数据的来源、许可和目录约定见：

- [data/README.md](data/README.md)
- [dem/README.md](dem/README.md)
- [configs/README.md](configs/README.md)

当前入口脚本仍读取各自文件顶部的 `CONFIG` 字典；`configs/example.yaml` 是下一阶段配置化迁移的目标格式，当前不会被旧脚本自动读取。

### DEM

完整 1 m DEM 不进入 Git。纽约市 DEM 使用 `USGS/3DEP/1m`，通过 Earth Engine 导出到 Google Drive，再下载到本地 `dem/`。详细步骤见 [dem/README.md](dem/README.md)。

### Base-station points and boundaries

基站点数据可以公开发布，并通过 Git LFS 下载。Shapefile 的 `.shp`、`.shx`、`.dbf`、`.prj` 等组成文件必须保持同一目录，不要单独移动其中一个文件。

## Usage

### Export New York 1 m DEM

先完成 Earth Engine 授权并确认项目 ID：

```powershell
earthengine authenticate --auth_mode=localhost
```

先做不提交任务的检查：

```powershell
python download_newyork_1m_dem.py `
  --project gen-lang-client-0153149292 `
  --dry-run
```

确认边界、分辨率和 Google Drive 文件夹后提交导出：

```powershell
python download_newyork_1m_dem.py `
  --project gen-lang-client-0153149292 `
  --export
```

脚本默认使用纽约市五县联合边界，输出 EPSG:3857、1 m、分片 GeoTIFF。也可以直接在 GEE Code Editor 中运行 `gee_newyork_1m_dem.js`。

### Select a scenario

准备边界、基站点和本地 DEM 后运行：

```powershell
python select_sites.py
```

脚本当前的城市、DEM 和扫描参数在 `select_sites.py` 顶部 `CONFIG` 中设置。

### Generate figures from an existing CSV

```powershell
python generate_scenario_figures.py --help
python generate_scenario_figures.py --size 3000 --target 30
```

该脚本读取已有 CSV，并根据 `CONFIG` 中的 DEM 路径生成三维图。

## Reproducibility

每次正式实验应记录：输入数据来源和 SHA256、边界名称与 CRS、DEM 分辨率和 CRS、扫描参数、软件版本、输出文件和随机种子。运行模板位于 `runs/templates/`，已完成实验只提交简短摘要到 `runs/summaries/`。

## Licensing and attribution

源代码采用 MIT License，见 [LICENSE](LICENSE)。数据不自动继承代码许可证：基站点和边界数据按已确认的再分发授权及其元数据发布，USGS 3DEP 数据按 USGS 和 Earth Engine 数据目录要求进行引用。

## Known limitations

- 当前两个本地场景脚本仍然包含重复代码和顶部硬编码配置；
- 当前交互式场景选择依赖桌面图形环境；
- 1 m 城市 DEM 体量很大，不适合放入普通 Git 历史；
- CI 将只使用小型 fixture，不会访问 Earth Engine 或下载完整 DEM。
