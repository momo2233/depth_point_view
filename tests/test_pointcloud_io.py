from __future__ import annotations

import numpy as np
import pytest

from depth_viewer.models import PointCloud
from depth_viewer.ply import encode_binary_ply
from depth_viewer.pointcloud_io import load_ply_bytes, sample_point_cloud


def test_load_exported_binary_ply_with_colors() -> None:
    source = PointCloud(
        points=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        colors=np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
    )

    loaded = load_ply_bytes(encode_binary_ply(source), "cloud.ply")

    np.testing.assert_allclose(loaded.cloud.points, source.points)
    np.testing.assert_array_equal(loaded.cloud.colors, source.colors)
    assert loaded.warnings == ()


def test_load_ascii_ply_without_colors_and_remove_invalid_points() -> None:
    data = b"""ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
end_header
0 1 2
3 4 5
nan 1 2
"""

    loaded = load_ply_bytes(data, "cloud.ply")

    assert loaded.cloud.points.shape == (2, 3)
    assert loaded.cloud.colors.shape == (2, 3)
    assert any("默认蓝色" in warning for warning in loaded.warnings)
    assert any("1 个" in warning for warning in loaded.warnings)


def test_load_ply_supports_short_rgb_names_and_float_colors() -> None:
    data = b"""ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
property float r
property float g
property float b
end_header
1 2 3 1.0 0.5 0.0
"""

    loaded = load_ply_bytes(data, "cloud.ply")

    np.testing.assert_array_equal(loaded.cloud.colors[0], [255, 128, 0])


def test_reject_ply_without_xyz() -> None:
    data = b"""ply
format ascii 1.0
element vertex 1
property float x
property float y
end_header
1 2
"""
    with pytest.raises(ValueError, match="x、y、z"):
        load_ply_bytes(data, "cloud.ply")


def test_uploaded_cloud_sampling_is_deterministic_and_limited() -> None:
    cloud = PointCloud(
        points=np.arange(3000, dtype=np.float32).reshape(1000, 3),
        colors=np.zeros((1000, 3), dtype=np.uint8),
    )

    sampled = sample_point_cloud(cloud, 100)

    assert sampled.points.shape == (100, 3)
    np.testing.assert_array_equal(sampled.points[0], cloud.points[0])
    np.testing.assert_array_equal(sampled.points[-1], cloud.points[-1])
