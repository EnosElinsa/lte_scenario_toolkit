# Experiment configurations

`example.yaml` 是后续配置化迁移的目标格式，当前仍属于声明性示例；现有 `select_sites.py` 和 `generate_scenario_figures.py` 继续读取各自文件顶部的 `CONFIG` 字典。

字段对应关系：

| YAML 字段 | 当前脚本配置 | 含义 |
|---|---|---|
| `inputs.points_root` | `points_root` | 基站点数据根目录 |
| `inputs.points_layer` | `points_layer` | 基站点图层目录和名称 |
| `inputs.boundary_root` | `boundary_root` | 行政边界根目录 |
| `inputs.city` | `city_id`/自动发现 | 当前城市选择；后续改为名称优先 |
| `inputs.dem_path` | `dem_path` | 本地 DEM 路径 |
| `spatial.target_crs` | 固定 `EPSG:3857` | 分析坐标系 |
| `spatial.rectangle_size_m` | `rect_size` | 候选矩形边长 |
| `spatial.target_base_station_count` | `target_count` | 目标基站数量 |
| `spatial.count_tolerance` | `tolerance` | 数量容差 |
| `scan.strategy` | `strategy` | 矩形扫描策略 |
| `scan.step_m` | `scan_step` | 扫描步长 |
| `scan.max_rectangles` | `max_rects` | 最大候选矩形数 |
| `scan.minimum_center_spacing_m` | `min_spacing` | 候选中心最小间距 |
| `outputs.root` | `output_root` | 结果根目录 |

后续配置化任务会让脚本通过 `--config configs/example.yaml` 读取该文件，并保留旧的顶部配置作为兼容入口。
