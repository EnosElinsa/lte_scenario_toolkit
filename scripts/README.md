# Command-line entry points

- `select_sites.py`：转发到配置化场景选择入口；
- `generate_scenario_figures.py`：转发到已有 CSV 的论文图入口；
- `download_newyork_1m_dem.py`：转发到 GEE 导出入口；
- `create_data_manifest.py`：根据 `data/datasets.yaml` 生成文件大小和 SHA256 清单。

这些文件只负责把仓库的 `src/` 加入模块搜索路径并调用 `lte_scenario_toolkit` 包内 `main()`，不包含业务算法。
