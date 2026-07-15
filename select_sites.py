"""
在城市行政边界内扫描候选矩形 → 交互选择 → 提取基站点+DEM高程 → 导出CSV + 3D地形图
依赖: geopandas, shapely, numpy, pandas, rasterio, matplotlib, plotly
安装: pip install geopandas shapely numpy pandas rasterio matplotlib plotly
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
from shapely.geometry import box
from shapely.prepared import prep
import json
import time
import warnings

from src.config import load_experiment_config
from src import io as reproducible_io
from src import scenario as scenario_core
from src import spatial as spatial_core
from src import terrain as terrain_core
from src import visualization as visualization_core

warnings.filterwarnings('ignore')

# ---- 中文字体 ----
for _font in ['Microsoft YaHei', 'SimHei', 'STSong']:
    try:
        mpl.font_manager.FontProperties(family=_font).get_name()
        mpl.rcParams['font.sans-serif'] = [_font, 'DejaVu Sans']
        mpl.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent


# ======================== 参数配置 ========================
CONFIG = {
    # --- 输入输出 ---
    "points_root":   r".\points_shp",              # 点数据根目录
    "points_layer":  "USA_Clear_LTE_Base_Station", # 点图层名 = 文件夹名 = shp文件名
    "boundary_root": r".\boundary_shp",            # 边界数据根目录
    "city_id":       2,                            # 城市编号, 按 boundary_root 子目录名排序, 从1开始
    "output_root":   r".\results",                 # 输出根目录

    # --- DEM ---
    "dem_path":      r".\dem\USGS_1M_DEM_Chicago\USGS_1M_DEM_Chicago.tif",

    # --- 矩形与目标 ---
    "rect_size":     3000,      # 矩形边长 (米, Web Mercator坐标单位)
    "target_count":  30,        # 目标包含点数
    "tolerance":     0,         # 容差

    # --- 扫描控制 ---
    "scan_step":     10,         # 滑动步长(米), 越小越精细但越慢
    "max_rects":     100,       # 最大输出矩形数量
    "min_spacing":   3000,       # 结果矩形中心最小间距(米), 避免大量重叠

    # --- 策略 ---
    "strategy":      "uniform", # "sequential" 顺序扫描 | "uniform" 均匀分布扫描
}


# ======================== 核心函数 ========================

def resolve_root_dir(root_dir):
    """将相对目录解析为相对于脚本所在目录的绝对路径"""
    root = Path(root_dir)
    if not root.is_absolute():
        root = SCRIPT_DIR / root
    return root.resolve()


def build_layer_shp_path(root_dir, layer_name):
    """根据目录结构定位shp路径, 优先匹配同名shp, 否则接受目录内唯一shp"""
    layer_dir = resolve_root_dir(root_dir) / layer_name
    exact_shp = layer_dir / f"{layer_name}.shp"
    if exact_shp.exists():
        return exact_shp

    shp_files = sorted(layer_dir.glob("*.shp"))
    if len(shp_files) == 1:
        return shp_files[0]

    return exact_shp


def discover_boundary_layers(boundary_root):
    """自动发现可用城市, 允许目录名和shp主文件名不同"""
    root = resolve_root_dir(boundary_root)
    if not root.exists():
        raise FileNotFoundError(f"边界根目录不存在: {root}")

    layers = []
    for folder in sorted(root.iterdir(), key=lambda p: p.name):
        if not folder.is_dir():
            continue
        shp_files = sorted(folder.glob("*.shp"))
        if len(shp_files) == 1:
            shp_path = shp_files[0]
            layers.append({
                "folder_name": folder.name,
                "layer_name": shp_path.stem,
                "shp_path": shp_path,
            })
            continue

        exact_shp = folder / f"{folder.name}.shp"
        if exact_shp.exists():
            layers.append({
                "folder_name": folder.name,
                "layer_name": exact_shp.stem,
                "shp_path": exact_shp,
            })

    if not layers:
        raise FileNotFoundError(f"未在 {root} 下找到可用城市边界")

    return layers


def resolve_io_paths(cfg):
    """解析输入输出路径, 并按统一规范组织输出文件"""
    boundary_layers = discover_boundary_layers(cfg["boundary_root"])
    city_id = cfg["city_id"]
    if city_id < 1 or city_id > len(boundary_layers):
        city_list = " | ".join(
            f"{idx}:{item['folder_name']}" for idx, item in enumerate(boundary_layers, start=1)
        )
        raise ValueError(
            f"city_id 超出范围: {city_id}. 可选城市: {city_list}"
        )

    boundary_item = boundary_layers[city_id - 1]
    boundary_folder = boundary_item["folder_name"]
    boundary_layer = boundary_item["layer_name"]
    points_path = build_layer_shp_path(cfg["points_root"], cfg["points_layer"])
    boundary_path = boundary_item["shp_path"]

    if not points_path.exists():
        raise FileNotFoundError(f"点数据不存在: {points_path}")
    if not boundary_path.exists():
        raise FileNotFoundError(f"边界数据不存在: {boundary_path}")

    city_tag = boundary_folder
    base_name = (
        f"{city_tag}_{cfg['rect_size']}m_"
        f"target{cfg['target_count']}_tol{cfg['tolerance']}"
    )
    cache_name = (
        f"{city_tag}_{cfg['rect_size']}m_"
        f"target{cfg['target_count']}_tol{cfg['tolerance']}_"
        f"step{cfg['scan_step']}_sp{cfg['min_spacing']}_{cfg['strategy']}"
    )
    output_dir = resolve_root_dir(cfg["output_root"]) / city_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # 解析 DEM 路径 (相对路径基于脚本所在目录)
    dem_path = resolve_root_dir(cfg["dem_path"])

    return {
        "city_id": city_id,
        "city_options": [item["folder_name"] for item in boundary_layers],
        "boundary_folder": boundary_folder,
        "boundary_layer": boundary_layer,
        "points_shp": points_path,
        "boundary_shp": boundary_path,
        "dem_path": dem_path,
        "output_dir": output_dir,
        "output_csv": output_dir / f"{base_name}.csv",
        "output_3d_png": output_dir / f"{base_name}_3d.png",
        "output_3d_html": output_dir / f"{base_name}_3d.html",
        "preview_png": output_dir / f"{base_name}.png",
        "cache_json": output_dir / f"{cache_name}_cache.json",
    }


def load_and_prepare(cfg):
    """加载数据并预处理"""
    print("📂 加载数据...")

    points_gdf = gpd.read_file(cfg["points_shp"])
    boundary_gdf = gpd.read_file(cfg["boundary_shp"])

    if points_gdf.crs is None:
        raise ValueError(f"点数据缺少坐标系: {cfg['points_shp']}")

    # 统一投影到 EPSG:3857 (Web Mercator, 单位: 米), 保证 rect_size/scan_step 正确
    target_epsg = 3857
    if points_gdf.crs.to_epsg() != target_epsg:
        print(f"   点数据坐标系 {points_gdf.crs} -> 投影到 EPSG:{target_epsg}")
        points_gdf = points_gdf.to_crs(epsg=target_epsg)
    if boundary_gdf.crs.to_epsg() != target_epsg:
        print(f"   边界坐标系 {boundary_gdf.crs} -> 投影到 EPSG:{target_epsg}")
        boundary_gdf = boundary_gdf.to_crs(epsg=target_epsg)

    boundary = boundary_gdf.unary_union

    # 只保留边界内的点(加速后续计算)
    inside_mask = points_gdf.within(boundary)
    points_gdf = points_gdf[inside_mask].copy().reset_index(drop=True)

    # 提取坐标为numpy数组(核心加速手段)
    xs = np.asarray(points_gdf.geometry.x.to_numpy(), dtype=float)
    ys = np.asarray(points_gdf.geometry.y.to_numpy(), dtype=float)
    coords = np.column_stack((xs, ys))

    print(f"   基站点数: {len(coords)}")
    print(f"   坐标系  : {points_gdf.crs}")
    print(f"   边界范围: {boundary.bounds}")

    return points_gdf, boundary, coords


def generate_scan_positions(boundary, rect_size, step, strategy="sequential"):
    """
    生成扫描位置
    strategy:
      - "sequential": 从左下到右上顺序扫描
      - "uniform":    随机打乱, 获得空间均匀分布的结果
    """
    bminx, bminy, bmaxx, bmaxy = boundary.bounds

    xs = np.arange(bminx, bmaxx - rect_size, step)
    ys = np.arange(bminy, bmaxy - rect_size, step)

    # 生成所有(x, y)组合
    xx, yy = np.meshgrid(xs, ys)
    positions = np.column_stack([xx.ravel(), yy.ravel()])

    if strategy == "uniform":
        np.random.seed(42)
        np.random.shuffle(positions)

    print(f"   扫描策略  : {strategy}")
    print(f"   候选位置数: {len(positions)}")

    return positions


def scan_rectangles(coords, boundary, positions, cfg):
    """
    核心扫描函数
    使用NumPy向量化实现高速点计数
    """
    rect_size = cfg["rect_size"]
    target_min = cfg["target_count"] - cfg["tolerance"]
    target_max = cfg["target_count"] + cfg["tolerance"]
    max_rects  = cfg["max_rects"]
    min_spacing = cfg["min_spacing"]

    prep_boundary = prep(boundary)

    results = []
    centers_list = []
    t0 = time.time()

    total = len(positions)

    for i, (x, y) in enumerate(positions):
        # ---- 进度打印 ----
        if (i + 1) % 50000 == 0:
            pct = (i + 1) / total * 100
            elapsed = time.time() - t0
            print(f"   进度: {pct:.1f}% ({i+1}/{total}) | "
                  f"已找到: {len(results)} | 耗时: {elapsed:.1f}s")

        # ---- Step1: NumPy快速计数(最快的筛选) ----
        mask = (
            (coords[:, 0] >= x) & (coords[:, 0] <= x + rect_size) &
            (coords[:, 1] >= y) & (coords[:, 1] <= y + rect_size)
        )
        pt_count = mask.sum()

        if pt_count < target_min or pt_count > target_max:
            continue

        # ---- Step2: 与已有结果的间距检查(避免聚集) ----
        cx = x + rect_size / 2
        cy = y + rect_size / 2

        if centers_list:
            centers = np.array(centers_list)
            dists = np.sqrt((centers[:, 0] - cx)**2 + (centers[:, 1] - cy)**2)
            if np.any(dists < min_spacing):
                continue

        # ---- Step3: 边界包含检查(最慢, 放最后) ----
        rect_geom = box(x, y, x + rect_size, y + rect_size)
        if not prep_boundary.contains(rect_geom):
            continue

        # ---- 记录结果 ----
        results.append({
            'geometry':  rect_geom,
            'pt_count':  int(pt_count),
            'left_x':    round(x, 2),
            'bottom_y':  round(y, 2),
            'center_x':  round(cx, 2),
            'center_y':  round(cy, 2),
        })
        centers_list.append([cx, cy])

        print(f"   ✅ 第{len(results):>3d}个 | 中心({cx:.0f}, {cy:.0f}) | "
              f"点数={pt_count}")

        if len(results) >= max_rects:
            print(f"   🎯 已达最大数量 {max_rects}, 停止扫描")
            break

    elapsed = time.time() - t0
    print(f"   扫描完成: 找到 {len(results)} 个矩形, 总耗时 {elapsed:.1f}s")

    return results


def verify_results(results, coords, rect_size):
    """验证结果正确性"""
    print("\n🔍 验证结果...")
    for i, r in enumerate(results):
        x, y = r['left_x'], r['bottom_y']
        mask = (
            (coords[:, 0] >= x) & (coords[:, 0] <= x + rect_size) &
            (coords[:, 1] >= y) & (coords[:, 1] <= y + rect_size)
        )
        actual = mask.sum()
        status = "✅" if actual == r['pt_count'] else "❌"
        if i < 5 or actual != r['pt_count']:  # 只打印前5个和异常的
            print(f"   {status} 矩形{i+1}: 记录={r['pt_count']}, 验证={actual}")
    print(f"   验证完成, 共 {len(results)} 个矩形")


def extract_elevation(points_gdf, dem):
    """从DEM中提取每个点的高程值, 自动处理坐标系转换"""
    if len(points_gdf) == 0:
        return np.array([])

    dem_crs = dem.crs
    if points_gdf.crs.to_epsg() != dem_crs.to_epsg():
        temp_gdf = points_gdf.to_crs(dem_crs)
        xs = temp_gdf.geometry.x.values
        ys = temp_gdf.geometry.y.values
    else:
        xs = points_gdf.geometry.x.values
        ys = points_gdf.geometry.y.values

    dem_band = dem.read(1)
    nodata = dem.nodata
    elevations = []

    for x, y in zip(xs, ys):
        try:
            row, col = dem.index(x, y)
            if 0 <= row < dem.height and 0 <= col < dem.width:
                val = dem_band[row, col]
                if nodata is not None and val == nodata:
                    elevations.append(np.nan)
                else:
                    elevations.append(float(val))
            else:
                elevations.append(np.nan)
        except Exception:
            elevations.append(np.nan)

    return np.array(elevations)


def process_selected_rectangles(chosen, points_gdf, dem, cfg):
    """
    对每个选中的矩形: 裁剪点 → 赋属性 → 提取高程 → 构建输出行
    返回 (final_df, selected_points_gdf)
      - final_df: 合并后的输出 DataFrame
      - selected_points_gdf: 裁剪后的 GeoDataFrame (含 elevation 列, 用于3D渲染)
    """
    if not chosen:
        print("\n⚠ 没有选中任何矩形! 建议:")
        print("   1. 增大 tolerance (容差)")
        print("   2. 减小 scan_step (步长)")
        print("   3. 在交互窗口中点击选择矩形")
        return None, None

    rect_size = cfg["rect_size"]
    points_crs = points_gdf.crs
    all_dfs = []
    all_selected = []

    for i, r in enumerate(chosen):
        rect_id = i + 1
        rect_geom = r['geometry']
        print(f"\n   {'─' * 45}")
        print(f"   处理矩形 {rect_id}/{len(chosen)}  "
              f"(中心: {r['center_x']:.0f}, {r['center_y']:.0f}  点数: {r['pt_count']})")

        # 空间裁剪
        minx, miny, maxx, maxy = rect_geom.bounds
        rough_mask = (
            (points_gdf.geometry.x >= minx) & (points_gdf.geometry.x <= maxx) &
            (points_gdf.geometry.y >= miny) & (points_gdf.geometry.y <= maxy)
        )
        candidates = points_gdf[rough_mask]
        within_mask = candidates.geometry.within(rect_geom)
        selected = candidates[within_mask].copy().reset_index(drop=True)

        if len(selected) == 0:
            print(f"     该矩形内无基站点, 跳过")
            continue

        print(f"     裁剪到 {len(selected)} 个点")

        # 提取高程
        elevations = terrain_core.extract_elevation(selected, dem)
        terrain_core.require_valid_elevations(elevations)
        selected["elevation"] = elevations
        valid = np.sum(~np.isnan(elevations))
        elev_valid = elevations[~np.isnan(elevations)]
        if len(elev_valid) > 0:
            print(f"     高程: {valid}/{len(elevations)} 有效, "
                  f"[{elev_valid.min():.1f}, {elev_valid.max():.1f}]m")

        all_selected.append(selected)

        # 构建输出行
        df = reproducible_io.build_output_dataframe(
            selected, points_crs,
            rect_id=rect_id,
            pt_count=r['pt_count'],
            left_x=r['left_x'],
            bottom_y=r['bottom_y'],
            center_x=r['center_x'],
            center_y=r['center_y'],
            rect_size=rect_size,
        )
        all_dfs.append(df)

    if not all_dfs:
        return None, None

    final_df = pd.concat(all_dfs, ignore_index=True)
    merged_selected = gpd.GeoDataFrame(
        pd.concat(all_selected, ignore_index=True), crs=points_crs
    )
    return final_df, merged_selected


def build_output_dataframe(selected, points_crs, *,
                           rect_id, pt_count, left_x, bottom_y,
                           center_x, center_y, rect_size):
    """
    构建输出列:
    cell, lon, lat, range, X, Y, rect_id, pt_count,
    left_x, bottom_y, center_x, center_y, elevation
    """
    df = pd.DataFrame()

    # cell
    for col_name in ['cell', 'Cell', 'CELL']:
        if col_name in selected.columns:
            df['cell'] = selected[col_name].values
            break
    else:
        df['cell'] = range(1, len(selected) + 1)

    # lon, lat (WGS84)
    if points_crs.to_epsg() == 4326:
        df['lon'] = selected.geometry.x.values
        df['lat'] = selected.geometry.y.values
    else:
        wgs84 = selected.to_crs(epsg=4326)
        df['lon'] = wgs84.geometry.x.values
        df['lat'] = wgs84.geometry.y.values

    # range (从shapefile字段读取实际覆盖范围 R_j)
    for col_name in ['range', 'Range', 'RANGE']:
        if col_name in selected.columns:
            df['range'] = selected[col_name].values
            break
    else:
        df['range'] = np.nan

    # X, Y (EPSG:3857)
    if points_crs.to_epsg() == 3857:
        df['X'] = selected.geometry.x.values
        df['Y'] = selected.geometry.y.values
    else:
        proj = selected.to_crs(epsg=3857)
        df['X'] = proj.geometry.x.values
        df['Y'] = proj.geometry.y.values

    # 矩形属性
    df['rect_id'] = rect_id
    df['pt_count'] = pt_count
    df['left_x'] = left_x
    df['bottom_y'] = bottom_y
    df['center_x'] = center_x
    df['center_y'] = center_y

    # elevation
    if 'elevation' in selected.columns:
        df['elevation'] = selected['elevation'].values

    return df


# ======================== 主程序 ========================

def main(argv=None):
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Select an LTE scenario using a reproducible YAML configuration"
    )
    parser.add_argument("--config", type=Path, help="experiment YAML; paths are repository-relative")
    parser.add_argument("--city", help="boundary directory or layer name")
    parser.add_argument("--output-dir", type=Path, help="write outputs directly to this directory")
    parser.add_argument("--size", type=int, help="override rectangle size in metres")
    parser.add_argument("--target", type=int, help="override target base-station count")
    parser.add_argument(
        "--select-index",
        type=int,
        help="choose a one-based candidate index without opening the interactive selector",
    )
    args = parser.parse_args(argv)

    if args.config:
        cfg = load_experiment_config(
            args.config,
            city=args.city,
            output_dir=args.output_dir,
        )
    else:
        cfg = CONFIG.copy()
        cfg["repo_root"] = SCRIPT_DIR
        cfg["target_crs"] = "EPSG:3857"
        if args.city:
            cfg["city_name"] = args.city
        if args.output_dir:
            cfg["output_root"] = args.output_dir
            cfg["output_dir_is_final"] = True
    if args.size is not None:
        cfg["rect_size"] = args.size
    if args.target is not None:
        cfg["target_count"] = args.target

    print("=" * 60)
    print(f"  基站点数据 — {cfg['rect_size']}m×{cfg['rect_size']}m矩形框选工具")
    print("=" * 60)

    io_paths = spatial_core.resolve_io_paths(cfg)
    cfg.update(io_paths)

    city_list = " | ".join(
        f"{idx}:{name}" for idx, name in enumerate(cfg["city_options"], start=1)
    )
    print(f"   城市列表: {city_list}")
    print(f"   当前城市: {cfg['city_id']} -> {cfg['boundary_folder']} ({cfg['boundary_layer']})")
    print(f"   点数据  : {cfg['points_shp']}")
    print(f"   边界数据: {cfg['boundary_shp']}")
    print(f"   DEM数据 : {cfg['dem_path']}")
    print(f"   输出目录: {cfg['output_dir']}")

    # 1. 加载数据
    print(f"\n{'='*20} Step 1: 加载数据 {'='*20}")
    points_gdf, boundary, coords = spatial_core.load_and_prepare(cfg)

    # 2. 扫描 (或读取缓存)
    cache_path = cfg["cache_json"]
    if cache_path.exists():
        print(f"\n{'='*20} Step 2: 读取扫描缓存 {'='*20}")
        print(f"   缓存文件: {cache_path.name}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"   读取到 {len(results)} 个候选矩形")
        # 重建 geometry (缓存中不存 shapely 对象)
        for r in results:
            r['geometry'] = box(r['left_x'], r['bottom_y'],
                               r['left_x'] + cfg['rect_size'],
                               r['bottom_y'] + cfg['rect_size'])
    else:
        print(f"\n{'='*20} Step 2: 生成扫描网格 {'='*20}")
        positions = scenario_core.generate_scan_positions(
            boundary,
            cfg["rect_size"],
            cfg["scan_step"],
            cfg["strategy"],
            random_seed=cfg.get("random_seed", 42),
        )

        print(f"\n{'='*20} Step 3: 扫描矩形 {'='*20}")
        print(f"   目标点数: {cfg['target_count']} ± {cfg['tolerance']} "
              f"(即 {cfg['target_count']-cfg['tolerance']}~"
              f"{cfg['target_count']+cfg['tolerance']})")
        results = scenario_core.scan_rectangles(coords, boundary, positions, cfg)

        # 保存缓存 (去掉不可序列化的 geometry)
        cache_data = [{k: v for k, v in r.items() if k != 'geometry'} for r in results]
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"   缓存已保存: {cache_path.name}")

    # 3. 验证
    scenario_core.verify_results(results, coords, cfg["rect_size"])

    # 4. 交互选择
    print(f"\n{'='*20} Step 4: 交互选择 {'='*20}")
    if args.select_index is None:
        chosen = visualization_core.interactive_select(points_gdf, boundary, results, cfg)
    else:
        chosen = scenario_core.choose_result(results, args.select_index)

    if not chosen:
        print("\n⚠ 未选中任何矩形, 退出")
        return None

    # 5. 加载DEM + 提取高程 + 导出CSV
    print(f"\n{'='*20} Step 5: 提取点数据 + DEM高程 → CSV {'='*20}")
    dem_path = cfg["dem_path"]
    try:
        terrain_core.validate_dem_path(dem_path)
    except FileNotFoundError:
        print(f"   ❌ DEM文件不存在: {dem_path}")
        return None
    dem = rasterio.open(dem_path)
    dem_crs = str(dem.crs)
    dem_resolution_m = float(abs(dem.res[0]))
    print(f"   DEM: {dem_path.name}  CRS={dem.crs}  分辨率={dem.res}")

    final_df, selected_gdf = process_selected_rectangles(chosen, points_gdf, dem, cfg)

    if final_df is None or len(final_df) == 0:
        dem.close()
        
        print("\n⚠ 没有提取到任何数据!")
        return None

    # 保存CSV
    output_csv = cfg["output_csv"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"\n   💾 CSV已保存: {output_csv}")
    print(f"      总记录数 : {len(final_df)}")

    # 统计
    avg_elev = final_df['elevation'].mean()
    print(f"   📊 点数={len(final_df)}, 平均高程={avg_elev:.1f}m")

    print(f"\n   📋 前5行:")
    print(final_df.head().to_string(index=False))

    # 6. 生成3D地形图 (DEM仍保持打开)
    print(f"\n{'='*20} Step 6: 生成3D地形图 {'='*20}")
    try:
        visualization_core.render_3d_terrain(chosen[0], selected_gdf, dem, cfg)
    except Exception as exc:
        print(f"   ⚠ 3D地形图生成失败: {exc}")
    finally:
        dem.close()

    # 7. 可选: 2D预览图
    if cfg.get("save_preview_png", True):
        try:
            preview_path = visualization_core.save_preview(points_gdf, boundary, chosen, cfg)
            print(f"   📊 预览图已保存: {preview_path}")
        except Exception as exc:
            print(f"   ⚠ 预览图生成失败: {exc}")

    # 8. 保存可复现运行记录；Shapefile 的所有同名组成文件均纳入校验。
    try:
        input_records = []
        for role, main_path, source, license_name in (
            (
                "base_station_points",
                cfg["points_shp"],
                "data/manifest.json",
                "Public redistribution permission confirmed by repository owner",
            ),
            (
                "administrative_boundary",
                cfg["boundary_shp"],
                "data/manifest.json",
                "Public redistribution permission confirmed by repository owner",
            ),
        ):
            for component in sorted(main_path.parent.glob(f"{main_path.stem}.*")):
                if component.is_file():
                    input_records.append(
                        reproducible_io.build_dataset_record(
                            component,
                            name=f"{role}:{component.suffix.lstrip('.')}",
                            source_url=source,
                            license_name=license_name,
                            crs=str(points_gdf.crs),
                        )
                    )
        input_records.append(
            reproducible_io.build_dataset_record(
                cfg["dem_path"],
                name="dem",
                source_url="https://developers.google.com/earth-engine/datasets/catalog/USGS_3DEP_1m",
                license_name="USGS public-domain data; retain source attribution",
                crs=dem_crs,
                resolution_m=dem_resolution_m,
            )
        )
        outputs = [
            path
            for path in (
                cfg["output_csv"],
                cfg["output_3d_png"],
                cfg["output_3d_html"],
                cfg["preview_png"],
            )
            if path.exists()
        ]
        run_record = reproducible_io.write_run_record(
            cfg["output_dir"],
            config=cfg,
            inputs=input_records,
            outputs=outputs,
            command=sys.argv if argv is None else ["select_sites.py", *argv],
            filename="run-select-sites.json",
        )
        print(f"   🧾 运行记录已保存: {run_record}")
    except Exception as exc:
        print(f"   ⚠ 运行记录生成失败: {exc}")

    print("\n✅ 全部完成!")
    return final_df


def interactive_select(points_gdf, boundary, results, coords, cfg):
    """
    交互式单选矩形 (一次只能选一个):
      - 工具栏的 放大镜/十字箭头 用于缩放/平移 (先放大到密集区域再选)
      - 工具栏回到 普通指针 后, 点击矩形选中 (自动取消之前选中的)
      - 关闭窗口后返回选中的结果 (0 或 1 个)
    """
    from matplotlib.patches import Rectangle as MplRect

    if not results:
        print("   没有候选矩形可供选择")
        return []

    current = [None]  # 当前选中的索引, None=未选
    patches = []
    rect_size = cfg["rect_size"]

    fig, ax = plt.subplots(1, 1, figsize=(14, 11))

    gpd.GeoSeries([boundary]).plot(ax=ax, facecolor='none',
                                   edgecolor='black', linewidth=1.5)
    points_gdf.plot(ax=ax, color='gray', markersize=0.5, alpha=0.3)

    for i, r in enumerate(results):
        p = MplRect((r['left_x'], r['bottom_y']), rect_size, rect_size,
                    facecolor='red', edgecolor='red', alpha=0.3, linewidth=1.5)
        ax.add_patch(p)
        patches.append(p)
        ax.annotate(f"{i+1}\n({r['pt_count']})",
                    xy=(r['center_x'], r['center_y']),
                    ha='center', va='center', fontsize=7, color='darkred',
                    fontweight='bold')

    ax.set_title(
        f"缩放/平移后回到指针模式, 点击选择1个矩形 | "
        f"共 {len(results)} 个候选 | 关闭窗口完成",
        fontsize=11
    )
    ax.set_aspect('equal')
    ax.autoscale_view()

    status_text = fig.text(0.5, 0.01, "未选择", ha='center', fontsize=11,
                           color='green', fontweight='bold')

    def update_status():
        if current[0] is not None:
            idx = current[0]
            r = results[idx]
            status_text.set_text(
                f"已选: #{idx+1}  (点数={r['pt_count']}, "
                f"中心={r['center_x']:.0f},{r['center_y']:.0f})")
        else:
            status_text.set_text("未选择")
        fig.canvas.draw_idle()

    def on_click(event):
        toolbar = fig.canvas.manager.toolbar
        if toolbar is not None and toolbar.mode != '':
            return
        if event.inaxes != ax:
            return
        cx, cy = event.xdata, event.ydata
        for idx in reversed(range(len(results))):
            r = results[idx]
            if (r['left_x'] <= cx <= r['left_x'] + rect_size and
                    r['bottom_y'] <= cy <= r['bottom_y'] + rect_size):
                # 取消之前选中的
                prev = current[0]
                if prev is not None and prev != idx:
                    patches[prev].set_facecolor('red')
                    patches[prev].set_edgecolor('red')
                    patches[prev].set_alpha(0.3)

                if current[0] == idx:
                    # 再次点击同一个 → 取消选择
                    patches[idx].set_facecolor('red')
                    patches[idx].set_edgecolor('red')
                    patches[idx].set_alpha(0.3)
                    current[0] = None
                else:
                    # 选中新矩形
                    patches[idx].set_facecolor('green')
                    patches[idx].set_edgecolor('green')
                    patches[idx].set_alpha(0.45)
                    current[0] = idx

                update_status()
                return

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    print("   提示: 先用工具栏 放大镜 缩放到密集区域, 再点指针回到选择模式")
    print("         点击矩形选中(绿色, 一次只能选1个), 关闭窗口完成")
    plt.show()

    if current[0] is not None:
        chosen = [results[current[0]]]
        print(f"   已选择: #{current[0]+1} (点数={chosen[0]['pt_count']})")
    else:
        chosen = []
        print("   未选择任何矩形")
    return chosen


def preview(points_gdf, boundary, chosen, cfg):
    """可视化预览(可选) - chosen 是选中矩形的 list[dict]"""
    from matplotlib.patches import Rectangle as MplRect

    rect_size = cfg["rect_size"]
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    gpd.GeoSeries([boundary]).plot(ax=ax, facecolor='none',
                                   edgecolor='black', linewidth=1.5)
    points_gdf.plot(ax=ax, color='gray', markersize=0.5, alpha=0.3)

    for i, r in enumerate(chosen):
        p = MplRect((r['left_x'], r['bottom_y']), rect_size, rect_size,
                    facecolor='green', edgecolor='green', alpha=0.35, linewidth=1.5)
        ax.add_patch(p)
        ax.annotate(f"{i+1}\n({r['pt_count']})",
                    xy=(r['center_x'], r['center_y']),
                    ha='center', va='center', fontsize=6, color='darkgreen')

    ax.set_title(
        f"{cfg['boundary_layer']} | {len(chosen)} selected "
        f"({cfg['rect_size']}m×{cfg['rect_size']}m)",
        fontsize=14
    )
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(cfg["preview_png"], dpi=150)
    plt.show()
    print(f"   📊 预览图已保存: {cfg['preview_png']}")


def render_3d_terrain(rect_info, selected_points, dem, cfg):
    """
    生成选中矩形区域的 3D DEM 地形图, 标注基站位置
    输出: matplotlib 静态图 (.png) + plotly 交互式图 (.html)
    """
    from rasterio.windows import from_bounds
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import plotly.graph_objects as go

    rect_size = cfg["rect_size"]
    left_x = rect_info['left_x']
    bottom_y = rect_info['bottom_y']
    right_x = left_x + rect_size
    top_y = bottom_y + rect_size

    # ---- 从 DEM 裁剪矩形区域 ----
    dem_crs_epsg = dem.crs.to_epsg()
    pts_crs_epsg = 3857  # 点数据已投影到 3857

    # 如果 DEM CRS 与 3857 不同, 需要转换矩形边界坐标到 DEM CRS
    if dem_crs_epsg != pts_crs_epsg:
        from pyproj import Transformer
        transformer = Transformer.from_crs(f"EPSG:{pts_crs_epsg}", dem.crs, always_xy=True)
        bl_x, bl_y = transformer.transform(left_x, bottom_y)
        tr_x, tr_y = transformer.transform(right_x, top_y)
    else:
        bl_x, bl_y = left_x, bottom_y
        tr_x, tr_y = right_x, top_y

    window = from_bounds(bl_x, bl_y, tr_x, tr_y, dem.transform)
    dem_clip = dem.read(1, window=window)
    win_transform = dem.window_transform(window)

    nrows, ncols = dem_clip.shape
    if nrows == 0 or ncols == 0:
        print("   ⚠ DEM 裁剪区域为空, 无法生成3D地形图")
        return

    # 处理 nodata
    nodata = dem.nodata
    if nodata is not None:
        dem_clip = np.where(dem_clip == nodata, np.nan, dem_clip)

    # 生成网格坐标 (DEM 像素中心)
    cols = np.arange(ncols)
    rows = np.arange(nrows)
    col_grid, row_grid = np.meshgrid(cols, rows)
    xs = win_transform.c + col_grid * win_transform.a + row_grid * win_transform.b
    ys = win_transform.f + col_grid * win_transform.d + row_grid * win_transform.e

    # 如果 DEM CRS != 3857, 把网格坐标转回 3857 方便与点坐标对齐
    if dem_crs_epsg != pts_crs_epsg:
        from pyproj import Transformer
        transformer_back = Transformer.from_crs(dem.crs, f"EPSG:{pts_crs_epsg}", always_xy=True)
        xs_flat, ys_flat = transformer_back.transform(xs.ravel(), ys.ravel())
        xs = xs_flat.reshape(xs.shape)
        ys = ys_flat.reshape(ys.shape)

    # 基站点坐标 (EPSG:3857) 和高程
    pt_xs = selected_points.geometry.x.values
    pt_ys = selected_points.geometry.y.values
    pt_zs = selected_points["elevation"].values if "elevation" in selected_points.columns else np.full(len(selected_points), np.nan)

    # 将坐标平移到以矩形左下角为原点的局部坐标 (米)
    xs_local = xs - left_x
    ys_local = ys - bottom_y
    pt_xs_local = pt_xs - left_x
    pt_ys_local = pt_ys - bottom_y

    # ======================== 垂直夸张系数 ========================
    z_range = np.nanmax(dem_clip) - np.nanmin(dem_clip)
    z_range = max(z_range, 0.1)                        # 避免除零
    vertical_exag = 5                                   # 夸张倍数(可调)
    z_aspect = vertical_exag * z_range / rect_size      # plotly 用

    # ======================== matplotlib 静态 3D 图 ========================
    print("   🎨 生成 matplotlib 3D 地形图...")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 对大栅格做降采样, 避免渲染过慢
    max_grid = 300
    step_r = max(1, nrows // max_grid)
    step_c = max(1, ncols // max_grid)
    xs_ds = xs_local[::step_r, ::step_c]
    ys_ds = ys_local[::step_r, ::step_c]
    zs_ds = dem_clip[::step_r, ::step_c]

    surf = ax.plot_surface(xs_ds, ys_ds, zs_ds,
                           cmap='RdYlGn_r', alpha=0.85,
                           rstride=1, cstride=1,
                           linewidth=0, antialiased=True)

    # 标注基站 (红色散点, 略高于地表)
    z_offset = (np.nanmax(dem_clip) - np.nanmin(dem_clip)) * 0.02
    valid_mask = ~np.isnan(pt_zs)
    ax.scatter(pt_xs_local[valid_mask],
               pt_ys_local[valid_mask],
               pt_zs[valid_mask] + z_offset,
               c='red', s=15, depthshade=True,
               label=f'基站 ({valid_mask.sum()}个)', zorder=5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('高程 (m)')
    ax.set_title(
        f"3D地形 — {rect_size}m×{rect_size}m | "
        f"基站{rect_info['pt_count']}个 | "
        f"高程 [{np.nanmin(dem_clip):.0f}, {np.nanmax(dem_clip):.0f}]m",
        fontsize=12
    )
    ax.set_box_aspect([1, 1, z_aspect])               # 动态垂直比例
    ax.legend(loc='upper left')
    fig.colorbar(surf, ax=ax, shrink=0.5, label='高程 (m)')

    png_path = cfg["output_3d_png"]
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"   💾 静态3D图已保存: {png_path}")

    # ======================== plotly 交互式 3D 图 ========================
    print("   🌐 生成 plotly 交互式 3D 地形图...")

    # plotly 也做降采样
    max_plotly = 200
    step_r2 = max(1, nrows // max_plotly)
    step_c2 = max(1, ncols // max_plotly)
    xs_p = xs_local[::step_r2, ::step_c2]
    ys_p = ys_local[::step_r2, ::step_c2]
    zs_p = dem_clip[::step_r2, ::step_c2]

    terrain_surface = go.Surface(
        x=xs_p, y=ys_p, z=zs_p,
        colorscale=[[0,'rgb(0,104,55)'],[0.2,'rgb(49,163,84)'],[0.35,'rgb(166,217,106)'],[0.5,'rgb(255,255,191)'],[0.65,'rgb(253,174,97)'],[0.8,'rgb(215,48,39)'],[1,'rgb(103,0,13)']],
        opacity=0.9,
        colorbar=dict(title='高程 (m)'),
        name='地形',
        hovertemplate='X: %{x:.0f}m<br>Y: %{y:.0f}m<br>高程: %{z:.1f}m<extra></extra>'
    )

    # 基站散点
    hover_texts = []
    for i in range(len(pt_xs)):
        cell_val = ""
        for col_name in ['cell', 'Cell', 'CELL']:
            if col_name in selected_points.columns:
                cell_val = str(selected_points[col_name].iloc[i])
                break
        hover_texts.append(
            f"基站: {cell_val}<br>高程: {pt_zs[i]:.1f}m"
            if not np.isnan(pt_zs[i]) else f"基站: {cell_val}<br>高程: N/A"
        )

    stations = go.Scatter3d(
        x=pt_xs_local[valid_mask],
        y=pt_ys_local[valid_mask],
        z=pt_zs[valid_mask] + z_offset,
        mode='markers',
        marker=dict(size=4, color='red', symbol='diamond'),
        name=f'基站 ({valid_mask.sum()}个)',
        text=[t for t, v in zip(hover_texts, valid_mask) if v],
        hoverinfo='text',
    )

    fig_plotly = go.Figure(data=[terrain_surface, stations])
    fig_plotly.update_layout(
        title=dict(text=(
            f"3D地形 — {rect_size}m×{rect_size}m | "
            f"基站{rect_info['pt_count']}个 | "
            f"高程 [{np.nanmin(dem_clip):.0f}, {np.nanmax(dem_clip):.0f}]m"
        )),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='高程 (m)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=z_aspect),
        ),
        width=1200, height=800,
    )

    html_path = str(cfg["output_3d_html"])
    fig_plotly.write_html(html_path, include_plotlyjs=True)
    print(f"   💾 交互式3D图已保存: {html_path}")


if __name__ == "__main__":
    main()
