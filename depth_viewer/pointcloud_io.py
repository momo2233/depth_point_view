from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyParseError

from .models import PointCloud, PointCloudBundle


DEFAULT_CLOUD_COLOR = np.array([45, 166, 255], dtype=np.uint8)


def _read_colors(vertex: np.ndarray) -> tuple[np.ndarray, bool]:
    names = set(vertex.dtype.names or ())
    if {"red", "green", "blue"}.issubset(names):
        fields = ("red", "green", "blue")
    elif {"r", "g", "b"}.issubset(names):
        fields = ("r", "g", "b")
    else:
        colors = np.tile(DEFAULT_CLOUD_COLOR, (len(vertex), 1))
        return colors, False

    raw = np.column_stack([vertex[field] for field in fields])
    values = np.asarray(raw, dtype=np.float64)
    finite = np.isfinite(values)
    if np.any(finite):
        finite_values = values[finite]
        if np.issubdtype(raw.dtype, np.floating) and (
            np.min(finite_values) >= 0 and np.max(finite_values) <= 1
        ):
            values *= 255.0
    values[~finite] = 0
    return np.clip(np.rint(values), 0, 255).astype(np.uint8), True


def load_ply_bytes(data: bytes, filename: str) -> PointCloudBundle:
    if Path(filename).suffix.lower() != ".ply":
        raise ValueError("直接上传点云目前仅支持 PLY 文件")
    try:
        ply = PlyData.read(io.BytesIO(data))
    except (PlyParseError, ValueError, TypeError, EOFError, OSError) as exc:
        raise ValueError(f"无法读取 PLY 文件：{exc}") from exc

    try:
        vertex = ply["vertex"].data
    except KeyError as exc:
        raise ValueError("PLY 文件缺少 vertex 元素") from exc
    names = set(vertex.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY 顶点必须包含 x、y、z 属性")
    if len(vertex) == 0:
        raise ValueError("PLY 文件中没有顶点")

    points = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(
        np.float32, copy=False
    )
    colors, has_colors = _read_colors(vertex)
    finite = np.all(np.isfinite(points), axis=1)
    removed = int(len(points) - np.count_nonzero(finite))
    points = np.ascontiguousarray(points[finite])
    colors = np.ascontiguousarray(colors[finite])
    if not len(points):
        raise ValueError("PLY 文件中没有有限的有效顶点")

    warnings: list[str] = []
    if removed:
        warnings.append(f"已忽略 {removed:,} 个包含 NaN/无穷值的顶点")
    if not has_colors:
        warnings.append("PLY 不含 RGB 颜色，已使用默认蓝色显示")
    ignored_elements = [element.name for element in ply.elements if element.name != "vertex"]
    if ignored_elements:
        warnings.append("已忽略非点元素：" + "、".join(ignored_elements))
    return PointCloudBundle(
        cloud=PointCloud(points=points, colors=colors), warnings=tuple(warnings)
    )


def sample_point_cloud(cloud: PointCloud, max_points: int) -> PointCloud:
    count = cloud.points.shape[0]
    if count <= max_points:
        return cloud
    indices = np.linspace(0, count - 1, max_points, dtype=np.int64)
    return PointCloud(
        points=np.ascontiguousarray(cloud.points[indices]),
        colors=np.ascontiguousarray(cloud.colors[indices]),
    )
