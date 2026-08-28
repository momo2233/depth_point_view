from __future__ import annotations

import io
import math

import numpy as np
from matplotlib import colormaps
from PIL import Image

from .models import Intrinsics, PointCloud


COLORMAPS: dict[str, str] = {
    "Turbo": "turbo",
    "Viridis": "viridis",
    "Plasma": "plasma",
    "Inferno": "inferno",
    "Magma": "magma",
    "Cividis": "cividis",
    "Jet": "jet",
    "Rainbow": "rainbow",
    "灰度": "gray",
}


def suggest_depth_scale(dtype: np.dtype) -> float:
    return 0.001 if np.issubdtype(dtype, np.integer) else 1.0


def depth_to_meters(depth: np.ndarray, scale: float) -> np.ndarray:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("深度倍率必须是有限正数")
    return np.asarray(depth, dtype=np.float64) * scale


def valid_depth_mask(
    depth_m: np.ndarray,
    min_depth: float | None = None,
    max_depth: float | None = None,
) -> np.ndarray:
    depth = np.asarray(depth_m)
    mask = np.isfinite(depth) & (depth > 0)
    if min_depth is not None:
        mask &= depth >= min_depth
    if max_depth is not None:
        mask &= depth <= max_depth
    return mask


def depth_statistics(depth_m: np.ndarray) -> dict[str, float | int]:
    mask = valid_depth_mask(depth_m)
    valid = np.asarray(depth_m)[mask]
    if not valid.size:
        return {"valid": 0, "invalid": int(np.asarray(depth_m).size)}
    return {
        "valid": int(valid.size),
        "invalid": int(np.asarray(depth_m).size - valid.size),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "p02": float(np.percentile(valid, 2)),
        "p98": float(np.percentile(valid, 98)),
    }


def suggest_display_range(depth_m: np.ndarray) -> tuple[float, float]:
    stats = depth_statistics(depth_m)
    if stats["valid"] == 0:
        raise ValueError("depth 中没有有效像素")
    lower = float(stats["p02"])
    upper = float(stats["p98"])
    if upper <= lower:
        upper = lower + max(abs(lower) * 1e-6, 1e-6)
    return lower, upper


def render_pseudocolor(
    depth_m: np.ndarray,
    vmin: float,
    vmax: float,
    colormap: str = "Turbo",
    reverse: bool = False,
) -> np.ndarray:
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        raise ValueError("伪彩最大深度必须大于最小深度")
    if colormap not in COLORMAPS:
        raise ValueError(f"未知配色方案：{colormap}")

    depth = np.asarray(depth_m, dtype=np.float64)
    valid = valid_depth_mask(depth)
    normalized = np.zeros(depth.shape, dtype=np.float64)
    normalized[valid] = np.clip((depth[valid] - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap_name = COLORMAPS[colormap] + ("_r" if reverse else "")
    rgb = np.asarray(colormaps[cmap_name](normalized)[..., :3] * 255.0, dtype=np.uint8)
    rgb[~valid] = 0
    return rgb


def encode_png(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(
        buffer, format="PNG", optimize=True
    )
    return buffer.getvalue()


def validate_intrinsics_size(
    intrinsics: Intrinsics, image_shape: tuple[int, int]
) -> None:
    height, width = image_shape
    if intrinsics.width is not None and intrinsics.width != width:
        raise ValueError(
            f"内参宽度为 {intrinsics.width}，但 depth 宽度为 {width}"
        )
    if intrinsics.height is not None and intrinsics.height != height:
        raise ValueError(
            f"内参高度为 {intrinsics.height}，但 depth 高度为 {height}"
        )


def _sample_mask(mask: np.ndarray, max_points: int | None) -> np.ndarray:
    if max_points is None:
        return mask
    if max_points <= 0:
        raise ValueError("max_points 必须大于 0")
    count = int(np.count_nonzero(mask))
    if count <= max_points:
        return mask

    step = max(2, int(math.ceil(math.sqrt(count / max_points))))
    rows, columns = np.indices(mask.shape)
    sampled = mask & (rows % step == 0) & (columns % step == 0)
    sampled_count = int(np.count_nonzero(sampled))
    if sampled_count == 0:
        flat_indices = np.flatnonzero(mask)
        keep = np.linspace(0, count - 1, max_points, dtype=np.int64)
        sampled = np.zeros(mask.size, dtype=bool)
        sampled[flat_indices[keep]] = True
        return sampled.reshape(mask.shape)
    if sampled_count > max_points:
        flat_indices = np.flatnonzero(sampled)
        keep = np.linspace(0, sampled_count - 1, max_points, dtype=np.int64)
        sampled = np.zeros(mask.size, dtype=bool)
        sampled[flat_indices[keep]] = True
        sampled = sampled.reshape(mask.shape)
    return sampled


def project_depth_to_point_cloud(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    intrinsics: Intrinsics,
    min_depth: float | None = None,
    max_depth: float | None = None,
    max_points: int | None = None,
) -> PointCloud:
    depth = np.asarray(depth_m, dtype=np.float64)
    color = np.asarray(rgb)
    if depth.ndim != 2:
        raise ValueError("depth 必须是二维数组")
    if color.shape != (*depth.shape, 3):
        raise ValueError("RGB 与 depth 尺寸必须完全一致")
    validate_intrinsics_size(intrinsics, depth.shape)
    if min_depth is not None and max_depth is not None and max_depth < min_depth:
        raise ValueError("点云最大深度不能小于最小深度")

    mask = valid_depth_mask(depth, min_depth, max_depth)
    mask = _sample_mask(mask, max_points)
    rows, columns = np.nonzero(mask)
    z = depth[rows, columns]
    x = (columns.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = (rows.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
    points = np.column_stack((x, y, z)).astype(np.float32, copy=False)
    colors = color[rows, columns].astype(np.uint8, copy=False)
    return PointCloud(points=np.ascontiguousarray(points), colors=np.ascontiguousarray(colors))
