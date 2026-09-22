from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from depth_viewer.models import Distortion, Intrinsics
from depth_viewer.processing import (
    COLORMAPS,
    depth_statistics,
    depth_to_meters,
    encode_png,
    project_depth_to_point_cloud,
    render_pseudocolor,
    suggest_depth_scale,
    suggest_display_range,
    undistort_normalized_points,
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


def test_undistort_normalized_points_inverts_radial_tangential_model() -> None:
    distortion = Distortion(
        k1=0.083156, k2=-0.111383, k3=0.046383,
        k4=0.001, k5=-0.0005, k6=0.0001,
        p1=-0.000647, p2=-0.000524,
    )
    x_true = np.array([0.0, 0.15, -0.45, 0.7])
    y_true = np.array([0.0, -0.2, 0.3, -0.4])
    r2 = x_true**2 + y_true**2
    radial = (
        1 + distortion.k1 * r2 + distortion.k2 * r2**2 + distortion.k3 * r2**3
    ) / (
        1 + distortion.k4 * r2 + distortion.k5 * r2**2 + distortion.k6 * r2**3
    )
    xd = x_true * radial + 2 * distortion.p1 * x_true * y_true + distortion.p2 * (r2 + 2 * x_true**2)
    yd = y_true * radial + distortion.p1 * (r2 + 2 * y_true**2) + 2 * distortion.p2 * x_true * y_true

    x_restored, y_restored = undistort_normalized_points(xd, yd, distortion)

    np.testing.assert_allclose(x_restored, x_true, atol=1e-10)
    np.testing.assert_allclose(y_restored, y_true, atol=1e-10)


def test_distortion_toggle_changes_geometry_but_preserves_depth_and_colors() -> None:
    depth = np.full((2, 3), 2.0)
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    intrinsics = Intrinsics(
        fx=2, fy=2, cx=0, cy=0,
        distortion=Distortion(k1=0.2),
    )

    plain = project_depth_to_point_cloud(depth, rgb, intrinsics)
    corrected = project_depth_to_point_cloud(
        depth, rgb, intrinsics, correct_distortion=True
    )

    assert not np.allclose(corrected.points[:, :2], plain.points[:, :2])
    np.testing.assert_array_equal(corrected.points[:, 2], plain.points[:, 2])
    np.testing.assert_array_equal(corrected.colors, plain.colors)
    np.testing.assert_allclose(corrected.points[0], plain.points[0])


def test_distortion_toggle_requires_calibration_coefficients() -> None:
    with pytest.raises(ValueError, match="color_distortion"):
        project_depth_to_point_cloud(
            np.ones((1, 1)), np.zeros((1, 1, 3), dtype=np.uint8),
            Intrinsics(1, 1, 0, 0), correct_distortion=True,
        )
