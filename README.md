# lte_scenario_toolkit

面向研究者的可复现实验工具：在美国城市行政边界内筛选 LTE 基站场景，采样 1 m DEM 高程，并输出 CSV、二维预览、三维地形图和机器可读运行记录。

项目以明确的 `lte_scenario_toolkit` Python 包提供配置化工作流。研究者可以使用安装后的 CLI，也可以运行 `scripts/` 下的薄入口。

## Workflow

```text
准备公开基站点和行政边界
→ 下载或放置本地 DEM
→ 校验数据清单、CRS 和分辨率
→ 读取 YAML 实验配置
→ 扫描满足点数与间距约束的候选矩形
→ 交互选择，或用固定候选编号复现选择
→ 采样基站高程
→ 输出 CSV、PNG/EPS/HTML 和 run JSON
```

## Repository layout

```text
boundary_shp/                 # 已获公开再分发权限的边界 Shapefile
points_shp/                   # 公开 LTE 基站点；大型组成文件由 Git LFS 管理
dem/                          # 本地 DEM；不进入 Git/LFS
configs/                      # Chicago 与 New York YAML 实验配置
data/datasets.yaml            # 数据集来源、许可和空间元数据
data/manifest.json            # 文件大小与 SHA256 清单
src/lte_scenario_toolkit/     # 唯一 Python 实现包
scripts/                      # 薄命令行入口与 manifest 生成器
gee/newyork_1m_dem.js         # GEE Code Editor 导出脚本
tests/fixtures/               # 无需完整 DEM/基站数据的小型公开 fixture
runs/                         # 只跟踪模板和简短摘要
```

## Installation

要求 Python 3.10 或更高版本。Windows PowerShell：

```powershell
git lfs install
git lfs pull

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

安装后可使用 `lte-select-sites`、`lte-generate-figures` 和 `lte-download-newyork-dem`，也可使用对应的 `python scripts/...` 命令。

Python 导入名与项目名一致：

```python
from lte_scenario_toolkit.config import load_experiment_config
from lte_scenario_toolkit.scenario import scan_rectangles
```

## Data preparation

基站点和边界数据随仓库分发。DEM 体量很大，只保存在本地，目录和下载方式见 [dem/README.md](dem/README.md)。数据来源与许可声明见 [data/README.md](data/README.md)。

纽约市 1 m DEM 的 GEE 检查与导出：

```powershell
python scripts/download_newyork_1m_dem.py `
  --project gen-lang-client-0153149292 `
  --dry-run

python scripts/download_newyork_1m_dem.py `
  --project gen-lang-client-0153149292 `
  --export
```

导出的分片下载后需合并为配置引用的单一 GeoTIFF：

```text
dem/USGS_1M_DEM_NewYorkState_NewYork/USGS_1M_DEM_NewYorkState_NewYork.tif
```

准备或修改输入后重新生成完整校验清单：

```powershell
python scripts/create_data_manifest.py
```

## Reproducible scenario selection

Chicago 示例：

```powershell
python scripts/select_sites.py --config configs/example.yaml
```

第一次可以在桌面窗口选择候选矩形。记录候选编号后，正式实验用 `--select-index` 跳过人工选择，使选择过程可复现：

```powershell
python scripts/select_sites.py `
  --config configs/example.yaml `
  --select-index 1
```

纽约市示例（需先准备并合并 DEM）：

```powershell
python scripts/select_sites.py `
  --config configs/newyork.yaml `
  --select-index 1
```

通用覆盖参数：

```powershell
python scripts/select_sites.py `
  --config configs/example.yaml `
  --city Chicago `
  --output-dir results/custom-run `
  --size 3000 `
  --target 30
```

每次成功运行会在输出目录写入 `run-select-sites.json`，包含完整配置、Git commit、输入 SHA256、软件版本和输出文件清单。

## Generate figures from an existing CSV

```powershell
python scripts/generate_scenario_figures.py `
  --config configs/example.yaml
```

该入口读取配置推导出的 CSV，生成论文风格 PNG/EPS 和交互 HTML，并写入 `run-generate-figures.json`。

## Tests and CI

本地验证：

```powershell
python -m ruff check src scripts tests
python -m pytest -q
python -m compileall -q src/lte_scenario_toolkit scripts
node --check gee/newyork_1m_dem.js
```

GitHub Actions 使用小型矢量与内存 DEM fixture，不下载完整数据，也不访问 Earth Engine。

## Licensing and attribution

源代码采用 [MIT License](LICENSE)。数据不自动继承代码许可证：基站点和边界按仓库所有者确认的公开再分发权限发布；USGS 3DEP 数据保留 USGS 来源说明。缺失的原始来源 URL 和获取日期在 `data/datasets.yaml` 中明确标记为空，不做推测。

## Known limitations

- 首次候选选择仍需要桌面图形环境；正式复现实验可改用 `--select-index`；
- 本地场景处理要求单一 GeoTIFF，GEE 产生的纽约 DEM 分片需先合并；
- EPSG:3857 适合本项目城市尺度的米制扫描，但不是保面积投影；
- 当前只提供源码安装和 GitHub 使用方式，尚未发布到 PyPI。
