from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from depth_viewer.models import Intrinsics
from depth_viewer.processing import (
    COLORMAPS,
    depth_statistics,
    depth_to_meters,
    encode_png,
    project_depth_to_point_cloud,
    render_pseudocolor,
    suggest_depth_scale,
    suggest_display_range,
    validate_intrinsics_size,
)


def test_depth_scale_suggestion_and_conversion() -> None:
    assert suggest_depth_scale(np.dtype(np.uint16)) == 0.001
    assert suggest_depth_scale(np.dtype(np.float32)) == 1.0
    converted = depth_to_meters(np.array([[1000]], dtype=np.uint16), 0.001)
    np.testing.assert_allclose(converted, [[1.0]])


def test_statistics_ignore_zero_negative_nan_and_inf() -> None:
    depth = np.array([[0.0, -1.0, np.nan], [np.inf, 1.0, 3.0]])
    stats = depth_statistics(depth)
    assert stats["valid"] == 2
    assert stats["invalid"] == 4
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0


def test_display_range_uses_percentiles_and_handles_constant_depth() -> None:
    depth = np.arange(1, 101, dtype=float).reshape(10, 10)
    lower, upper = suggest_display_range(depth)
    assert lower == pytest.approx(np.percentile(depth, 2))
    assert upper == pytest.approx(np.percentile(depth, 98))

    constant_lower, constant_upper = suggest_display_range(np.ones((2, 2)))
    assert constant_upper > constant_lower


@pytest.mark.parametrize("colormap", list(COLORMAPS))
def test_all_pseudocolor_palettes_and_invalid_pixels(colormap: str) -> None:
    depth = np.array([[0.0, 1.0], [2.0, np.nan]])
    colored = render_pseudocolor(depth, 1.0, 2.0, colormap, reverse=True)
    assert colored.shape == (2, 2, 3)
    assert colored.dtype == np.uint8
    np.testing.assert_array_equal(colored[0, 0], [0, 0, 0])
    np.testing.assert_array_equal(colored[1, 1], [0, 0, 0])


def test_png_encoding_is_lossless_and_readable() -> None:
    image = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    encoded = encode_png(image)
    with Image.open(io.BytesIO(encoded)) as decoded:
        np.testing.assert_array_equal(np.asarray(decoded), image)


def test_pinhole_projection_and_rgb_correspondence() -> None:
    depth = np.ones((2, 2), dtype=float)
    rgb = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
        dtype=np.uint8,
    )
    intrinsics = Intrinsics(fx=1, fy=1, cx=0, cy=0, width=2, height=2)

    cloud = project_depth_to_point_cloud(depth, rgb, intrinsics)

    np.testing.assert_allclose(
        cloud.points,
        [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
    )
    np.testing.assert_array_equal(cloud.colors, rgb.reshape(-1, 3))


def test_preview_sampling_is_deterministic_and_limited() -> None:
    depth = np.ones((100, 100), dtype=float)
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    intrinsics = Intrinsics(100, 100, 50, 50)

    first = project_depth_to_point_cloud(depth, rgb, intrinsics, max_points=500)
    second = project_depth_to_point_cloud(depth, rgb, intrinsics, max_points=500)

    assert first.points.shape[0] <= 500
    np.testing.assert_array_equal(first.points, second.points)


def test_preview_sampling_falls_back_when_grid_misses_sparse_pixels() -> None:
    depth = np.zeros((100, 100), dtype=float)
    depth[1::2, 1::2] = 1.0
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    intrinsics = Intrinsics(100, 100, 50, 50)

    cloud = project_depth_to_point_cloud(depth, rgb, intrinsics, max_points=1000)

    assert cloud.points.shape[0] == 1000


def test_projection_rejects_rgb_and_calibration_size_mismatch() -> None:
    depth = np.ones((2, 2), dtype=float)
    rgb = np.zeros((3, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="尺寸"):
        project_depth_to_point_cloud(depth, rgb, Intrinsics(1, 1, 0, 0))

    with pytest.raises(ValueError, match="内参宽度"):
        validate_intrinsics_size(Intrinsics(1, 1, 0, 0, width=3), (2, 2))
