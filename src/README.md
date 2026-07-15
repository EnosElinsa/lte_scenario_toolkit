# Reusable modules

本目录包含入口脚本实际调用的可测试模块：

- `config.py`：YAML 读取、字段校验、路径解析和 CLI 覆盖后的扁平配置；
- `spatial.py`：边界发现、城市选择、CRS 统一、点在边界内筛选和输出路径；
- `scenario.py`：确定性扫描网格、点数/间距/边界约束、候选验证和固定编号选择；
- `terrain.py`：DEM 文件校验、高程采样和有效高程检查；
- `io.py`：稳定 CSV schema、SHA256 数据清单、软件版本和运行记录；
- `visualization.py`：交互选择、二维预览、静态/交互三维地形图。

根目录脚本仍保留历史 API，但主执行路径已经调用这些模块。新增算法或 I/O 行为应优先写在 `src/`，并先在 `tests/` 中增加 fixture 测试。
