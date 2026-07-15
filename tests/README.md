# Tests

测试覆盖 YAML 配置、城市边界发现、CRS 转换、边界内点筛选、确定性矩形扫描、间距和点数约束、DEM 采样、NoData/缺失文件错误、CSV 字段、数据 manifest、运行记录、二维/三维输出及 CLI。

`fixtures/` 只含两个很小的公开 GeoJSON；栅格测试使用 Rasterio 内存 GeoTIFF。测试不读取完整 LTE 数据或 1 m DEM，不访问 Earth Engine。

```powershell
python -m ruff check src scripts tests
python -m pytest -q
```
