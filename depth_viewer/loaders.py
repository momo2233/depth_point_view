from __future__ import annotations

import io
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import yaml
from PIL import Image, UnidentifiedImageError

from .models import DepthBundle, Intrinsics, RGBBundle


DEPTH_EXTENSIONS = {".png", ".tif", ".tiff", ".npy", ".npz"}
RGB_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _suffix(filename: str) -> str:
    return Path(filename).suffix.lower()


def normalize_depth_array(array: np.ndarray, label: str = "depth") -> np.ndarray:
    result = np.asarray(array)
    if result.ndim == 3 and result.shape[-1] == 1:
        result = result[..., 0]
    if result.ndim != 2:
        raise ValueError(f"{label} 必须是二维单通道数组，当前形状为 {result.shape}")
    if result.size == 0:
        raise ValueError(f"{label} 不能为空")
    if not (
        np.issubdtype(result.dtype, np.integer)
        or np.issubdtype(result.dtype, np.floating)
    ):
        raise ValueError(f"{label} 必须为整型或浮点型，当前类型为 {result.dtype}")
    return np.ascontiguousarray(result)


def load_depth_bytes(data: bytes, filename: str) -> DepthBundle:
    suffix = _suffix(filename)
    if suffix not in DEPTH_EXTENSIONS:
        raise ValueError(f"不支持的 depth 文件格式：{suffix or '未知'}")

    warnings: list[str] = []
    try:
        if suffix == ".npy":
            array = np.load(io.BytesIO(data), allow_pickle=False)
            arrays = {"depth": normalize_depth_array(array)}
        elif suffix == ".npz":
            arrays = {}
            rejected: list[str] = []
            with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                for key in archive.files:
                    try:
                        arrays[key] = normalize_depth_array(archive[key], key)
                    except ValueError:
                        rejected.append(key)
            if not arrays:
                raise ValueError("NPZ 中没有可用的二维数值 depth 数组")
            if rejected:
                warnings.append("已忽略非二维数值数组：" + "、".join(rejected))
            if len(arrays) > 1:
                warnings.append("NPZ 中包含多个 depth 数组，请在页面中选择")
        elif suffix in {".tif", ".tiff"}:
            with tifffile.TiffFile(io.BytesIO(data)) as tif:
                if not tif.pages:
                    raise ValueError("TIFF 文件中没有图像页")
                array = tif.pages[0].asarray()
                if len(tif.pages) > 1:
                    warnings.append(f"该 TIFF 包含 {len(tif.pages)} 页，当前仅使用第一页")
            arrays = {"depth": normalize_depth_array(array)}
        else:
            with Image.open(io.BytesIO(data)) as image:
                array = np.asarray(image)
            arrays = {"depth": normalize_depth_array(array)}
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(("NPZ", "depth", "不支持")):
            raise
        raise ValueError(f"无法读取 depth 文件：{exc}") from exc

    return DepthBundle(arrays=arrays, warnings=tuple(warnings))


def load_rgb_bytes(data: bytes, filename: str) -> RGBBundle:
    suffix = _suffix(filename)
    if suffix not in RGB_EXTENSIONS:
        raise ValueError(f"不支持的 RGB 文件格式：{suffix or '未知'}")
    warnings: list[str] = []
    try:
        with Image.open(io.BytesIO(data)) as image:
            frame_count = getattr(image, "n_frames", 1)
            if frame_count > 1:
                warnings.append(f"该图像包含 {frame_count} 帧，当前仅使用第一帧")
            image.seek(0)
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"无法读取 RGB 文件：{exc}") from exc
    return RGBBundle(image=np.ascontiguousarray(rgb), warnings=tuple(warnings))


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} 必须是对象/映射")
    return value


def _lookup(primary: Mapping[str, Any], root: Mapping[str, Any], key: str) -> Any:
    if key in primary:
        return primary[key]
    return root.get(key)


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if result <= 0:
        raise ValueError(f"{name} 必须大于 0")
    return result


def _optional_positive_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数值") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} 必须是有限正数")
    return result


def intrinsics_from_mapping(data: Mapping[str, Any]) -> Intrinsics:
    root = _as_mapping(data, "内参文件")
    if isinstance(root.get("intrinsics"), Mapping):
        nested = root["intrinsics"]
    elif isinstance(root.get("color"), Mapping):
        nested = root["color"]
    else:
        nested = root
    primary = _as_mapping(nested, "intrinsics")

    matrix: Any = _lookup(primary, root, "K")
    if matrix is None:
        matrix = _lookup(primary, root, "matrix")
    if matrix is None:
        camera_matrix = _lookup(primary, root, "camera_matrix")
        if isinstance(camera_matrix, Mapping):
            matrix = camera_matrix.get("data")
    if matrix is not None:
        try:
            matrix_array = np.asarray(matrix, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError("K/camera_matrix.data 必须是数值数组") from exc
        if matrix_array.size != 9:
            raise ValueError("K/camera_matrix.data 必须包含 9 个数值")
        matrix_values = {
            "fx": matrix_array[0],
            "fy": matrix_array[4],
            "cx": matrix_array[2],
            "cy": matrix_array[5],
        }
    else:
        matrix_values = {}

    values: dict[str, float] = {}
    for key in ("fx", "fy", "cx", "cy"):
        raw = _lookup(primary, root, key)
        if raw is None:
            raw = matrix_values.get(key)
        if raw is None:
            raise ValueError(f"内参缺少 {key}")
        try:
            values[key] = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} 必须是数值") from exc

    return make_intrinsics(
        **values,
        width=_optional_int(_lookup(primary, root, "width"), "width"),
        height=_optional_int(_lookup(primary, root, "height"), "height"),
        depth_scale=_optional_positive_float(
            _lookup(primary, root, "depth_scale"), "depth_scale"
        ),
    )


def make_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int | None = None,
    height: int | None = None,
    depth_scale: float | None = None,
) -> Intrinsics:
    values = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    for key, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{key} 必须是有限数值")
    if fx <= 0 or fy <= 0:
        raise ValueError("fx 和 fy 必须大于 0")
    return Intrinsics(
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        width=width,
        height=height,
        depth_scale=depth_scale,
    )


def load_intrinsics_bytes(data: bytes, filename: str) -> Intrinsics:
    suffix = _suffix(filename)
    if suffix not in {".json", ".yaml", ".yml"}:
        raise ValueError("内参文件仅支持 JSON、YAML 或 YML")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("内参文件必须使用 UTF-8 编码") from exc
    try:
        parsed = json.loads(text) if suffix == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"无法解析内参文件：{exc}") from exc
    return intrinsics_from_mapping(_as_mapping(parsed, "内参文件"))
