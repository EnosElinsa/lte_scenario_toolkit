# Reusable modules

`lte_scenario_toolkit/` 是安装后的同名 Python 包，也是项目唯一实现目录：

- `config.py`：YAML 读取、字段校验、路径解析和 CLI 覆盖后的扁平配置；
- `spatial.py`：边界发现、城市选择、CRS 统一、点在边界内筛选和输出路径；
- `scenario.py`：确定性扫描网格、点数/间距/边界约束、候选验证和固定编号选择；
- `terrain.py`：DEM 文件校验、高程采样和有效高程检查；
- `io.py`：稳定 CSV schema、SHA256 数据清单、软件版本和运行记录；
- `visualization.py`：交互选择、二维预览、静态/交互三维地形图。
- `select_sites.py`：候选扫描、选择、高程增强和运行记录编排；
- `generate_figures.py`：从已有场景 CSV 生成论文图；
- `newyork_dem.py`：纽约市 USGS 3DEP 1 m Earth Engine 导出。

`src/` 根目录不再作为名为 `src` 的 Python 包。新增行为应写入 `lte_scenario_toolkit/`，并先在 `tests/` 中增加 fixture 测试。
