# Experiment configurations

YAML 是本地场景工具的正式配置入口：

- `example.yaml`：Chicago 示例；
- `newyork.yaml`：纽约市五县边界示例，要求先准备合并后的本地 DEM。

```powershell
python scripts/select_sites.py --config configs/example.yaml
python scripts/generate_scenario_figures.py --config configs/example.yaml
```

字段映射：

| YAML 字段 | 运行字段 | 含义 |
|---|---|---|
| `experiment.name` | `experiment_name` | 运行名称 |
| `experiment.random_seed` | `random_seed` | `uniform` 扫描的确定性随机种子 |
| `inputs.points_root` | `points_root` | 基站点根目录 |
| `inputs.points_layer` | `points_layer` | Shapefile 目录和图层名 |
| `inputs.boundary_root` | `boundary_root` | 边界根目录 |
| `inputs.city` | `city_name` | 边界目录名或图层名，不区分大小写 |
| `inputs.dem_path` | `dem_path` | 单一 GeoTIFF 路径 |
| `spatial.target_crs` | `target_crs` | 分析坐标系，当前推荐 `EPSG:3857` |
| `spatial.rectangle_size_m` | `rect_size` | 候选矩形边长 |
| `spatial.target_base_station_count` | `target_count` | 目标基站数 |
| `spatial.count_tolerance` | `tolerance` | 点数容差 |
| `scan.*` | 扫描控制字段 | 策略、步长、最大候选数和中心间距 |
| `outputs.root` | `output_root` | 本次运行的最终输出目录 |

`--config` 是场景选择和图形生成的必需参数。命令行中的 `--city`、`--output-dir`、`--size` 和 `--target` 优先于 YAML。

`--select-index N` 使用一基编号固定选择第 N 个候选矩形，适合正式复现实验；未提供时打开交互选择窗口。
