from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Distortion:
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    model: str | int | None = None

    @property
    def is_zero(self) -> bool:
        return not any(
            (self.k1, self.k2, self.k3, self.k4, self.k5, self.k6, self.p1, self.p2)
        )


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int | None = None
    height: int | None = None
    depth_scale: float | None = None
    distortion: Distortion | None = None


@dataclass(frozen=True)
class DepthBundle:
    arrays: dict[str, np.ndarray]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RGBBundle:
    image: np.ndarray
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PointCloudBundle:
    cloud: "PointCloud"
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PointCloud:
    points: np.ndarray
    colors: np.ndarray

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("点坐标必须为 N×3 数组")
        if self.colors.shape != self.points.shape:
            raise ValueError("颜色数组必须与点坐标形状一致")
