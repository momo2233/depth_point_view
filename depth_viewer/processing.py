from __future__ import annotations

import io
import math

import numpy as np
from matplotlib import colormaps
from PIL import Image

from .models import Distortion, Intrinsics, PointCloud


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
    p02, p98 = np.percentile(valid, [2, 98])
    return {
        "valid": int(valid.size),
        "invalid": int(np.asarray(depth_m).size - valid.size),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "p02": float(p02),
        "p98": float(p98),
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
        buffer, format="PNG", optimize=False, compress_level=1
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
    row_grid = (np.arange(mask.shape[0]) % step == 0)[:, None]
    column_grid = (np.arange(mask.shape[1]) % step == 0)[None, :]
    sampled = mask & row_grid & column_grid
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


def undistort_normalized_points(
    x_distorted: np.ndarray,
    y_distorted: np.ndarray,
    distortion: Distortion,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the OpenCV radial-tangential model for normalized pixel rays."""
    xd = np.asarray(x_distorted, dtype=np.float64)
    yd = np.asarray(y_distorted, dtype=np.float64)
    if xd.shape != yd.shape:
        raise ValueError("畸变矫正的 X/Y 像素数组形状必须一致")
    if distortion.is_zero:
        return xd, yd

    x = xd.copy()
    y = yd.copy()
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(10):
            radius2 = x * x + y * y
            radius4 = radius2 * radius2
            radius6 = radius4 * radius2
            numerator = (
                1.0
                + distortion.k1 * radius2
                + distortion.k2 * radius4
                + distortion.k3 * radius6
            )
            denominator = (
                1.0
                + distortion.k4 * radius2
                + distortion.k5 * radius4
                + distortion.k6 * radius6
            )
            tangential_x = (
                2.0 * distortion.p1 * x * y
                + distortion.p2 * (radius2 + 2.0 * x * x)
            )
            tangential_y = (
                distortion.p1 * (radius2 + 2.0 * y * y)
                + 2.0 * distortion.p2 * x * y
            )
            inverse_radial = denominator / numerator
            x = (xd - tangential_x) * inverse_radial
            y = (yd - tangential_y) * inverse_radial
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("畸变矫正产生了无效坐标，请检查畸变模型和系数")
    return x, y


def project_depth_to_point_cloud(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    intrinsics: Intrinsics,
    min_depth: float | None = None,
    max_depth: float | None = None,
    max_points: int | None = None,
    correct_distortion: bool = False,
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
    distortion = intrinsics.distortion if correct_distortion else None
    if correct_distortion and distortion is None:
        raise ValueError("内参文件中没有 color_distortion，无法开启畸变矫正")
    points = np.empty((len(rows), 3), dtype=np.float32)
    # Keep temporary arrays small during full-resolution export and give
    # Python a chance to handle Ctrl+C between chunks.
    for start in range(0, len(rows), 250_000):
        end = min(start + 250_000, len(rows))
        z = depth[rows[start:end], columns[start:end]]
        xd = (columns[start:end].astype(np.float64) - intrinsics.cx) / intrinsics.fx
        yd = (rows[start:end].astype(np.float64) - intrinsics.cy) / intrinsics.fy
        if distortion is not None:
            x, y = undistort_normalized_points(xd, yd, distortion)
        else:
            x, y = xd, yd
        points[start:end, 0] = x * z
        points[start:end, 1] = y * z
        points[start:end, 2] = z
    colors = color[rows, columns].astype(np.uint8, copy=False)
    return PointCloud(points=np.ascontiguousarray(points), colors=np.ascontiguousarray(colors))
