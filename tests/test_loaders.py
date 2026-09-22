from __future__ import annotations

import io
import json

import numpy as np
import pytest
import tifffile
from PIL import Image

from depth_viewer.loaders import (
    intrinsics_from_mapping,
    load_depth_bytes,
    load_intrinsics_bytes,
    load_rgb_bytes,
    normalize_depth_array,
)


def test_load_uint16_png() -> None:
    source = np.array([[0, 1000], [2000, 65535]], dtype=np.uint16)
    buffer = io.BytesIO()
    Image.fromarray(source).save(buffer, format="PNG")

    loaded = load_depth_bytes(buffer.getvalue(), "depth.png")

    np.testing.assert_array_equal(loaded.arrays["depth"], source)
    assert loaded.arrays["depth"].dtype == np.uint16


def test_load_multipage_tiff_uses_first_page() -> None:
    first = np.full((2, 3), 100, dtype=np.uint16)
    second = np.full((2, 3), 200, dtype=np.uint16)
    buffer = io.BytesIO()
    with tifffile.TiffWriter(buffer) as writer:
        writer.write(first)
        writer.write(second)

    loaded = load_depth_bytes(buffer.getvalue(), "depth.tiff")

    np.testing.assert_array_equal(loaded.arrays["depth"], first)
    assert "2 页" in loaded.warnings[0]


def test_load_float_npy_and_squeeze_single_channel() -> None:
    source = np.arange(6, dtype=np.float32).reshape(2, 3, 1)
    buffer = io.BytesIO()
    np.save(buffer, source)

    loaded = load_depth_bytes(buffer.getvalue(), "depth.npy")

    assert loaded.arrays["depth"].shape == (2, 3)
    assert loaded.arrays["depth"].dtype == np.float32


def test_npz_lists_multiple_valid_arrays_and_skips_invalid() -> None:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        near=np.ones((2, 2), dtype=np.float32),
        far=np.ones((3, 4), dtype=np.uint16),
        metadata=np.arange(3),
    )

    loaded = load_depth_bytes(buffer.getvalue(), "capture.npz")

    assert list(loaded.arrays) == ["near", "far"]
    assert any("metadata" in warning for warning in loaded.warnings)
    assert any("多个" in warning for warning in loaded.warnings)


@pytest.mark.parametrize(
    "array",
    [np.empty((0, 3)), np.zeros((2, 2, 3)), np.array([[True]], dtype=bool)],
)
def test_reject_invalid_depth_arrays(array: np.ndarray) -> None:
    with pytest.raises(ValueError):
        normalize_depth_array(array)


def test_load_rgb_converts_to_uint8_rgb() -> None:
    source = Image.new("RGBA", (3, 2), (10, 20, 30, 100))
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")

    loaded = load_rgb_bytes(buffer.getvalue(), "rgb.png")

    assert loaded.image.shape == (2, 3, 3)
    assert loaded.image.dtype == np.uint8
    np.testing.assert_array_equal(loaded.image[0, 0], [10, 20, 30])


def test_intrinsics_support_flat_json() -> None:
    payload = {
        "width": 640,
        "height": 480,
        "fx": 615,
        "fy": 616,
        "cx": 319.5,
        "cy": 239.5,
        "depth_scale": 0.001,
    }

    intrinsics = load_intrinsics_bytes(json.dumps(payload).encode(), "camera.json")

    assert intrinsics.fx == 615
    assert intrinsics.width == 640
    assert intrinsics.depth_scale == 0.001


@pytest.mark.parametrize(
    "payload",
    [
        {"K": [600, 0, 320, 0, 601, 240, 0, 0, 1]},
        {"camera_matrix": {"data": [600, 0, 320, 0, 601, 240, 0, 0, 1]}},
        {"intrinsics": {"fx": 600, "fy": 601, "cx": 320, "cy": 240}},
    ],
)
def test_intrinsics_support_matrix_and_nested_forms(payload: dict) -> None:
    intrinsics = intrinsics_from_mapping(payload)
    assert (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy) == (
        600,
        601,
        320,
        240,
    )


def test_intrinsics_support_color_calibration_structure() -> None:
    payload = {
        "color": {
            "width": 3840,
            "height": 2160,
            "fx": 2254.25488281,
            "fy": 2253.15161133,
            "cx": 1930.19335938,
            "cy": 1078.02014160,
            "matrix": [
                [2254.25488281, 0.0, 1930.19335938],
                [0.0, 2253.15161133, 1078.02014160],
                [0.0, 0.0, 1.0],
            ],
        },
        "color_distortion": {
            "k1": 0.07992544,
            "k2": -0.10850055,
            "model": 4,
        },
    }

    intrinsics = intrinsics_from_mapping(payload)

    assert intrinsics.width == 3840
    assert intrinsics.height == 2160
    assert intrinsics.fx == pytest.approx(2254.25488281)
    assert intrinsics.fy == pytest.approx(2253.15161133)
    assert intrinsics.cx == pytest.approx(1930.19335938)
    assert intrinsics.cy == pytest.approx(1078.02014160)
    assert intrinsics.distortion is not None
    assert intrinsics.distortion.k1 == pytest.approx(0.07992544)
    assert intrinsics.distortion.k2 == pytest.approx(-0.10850055)
    assert intrinsics.distortion.model == 4


def test_intrinsics_parses_all_color_distortion_coefficients() -> None:
    payload = {
        "color": {"fx": 2257.163086, "fy": 2256.109619, "cx": 1908.697021, "cy": 1072.799805},
        "color_distortion": {
            "k1": 0.083156, "k2": -0.111383, "k3": 0.046383,
            "k4": 0.0, "k5": 0.0, "k6": 0.0,
            "p1": -0.000647, "p2": -0.000524, "model": None,
        },
    }

    intrinsics = intrinsics_from_mapping(payload)

    assert intrinsics.distortion is not None
    assert intrinsics.distortion.k3 == pytest.approx(0.046383)
    assert intrinsics.distortion.p1 == pytest.approx(-0.000647)
    assert intrinsics.distortion.p2 == pytest.approx(-0.000524)
    assert not intrinsics.distortion.is_zero


def test_intrinsics_rejects_nonfinite_distortion() -> None:
    with pytest.raises(ValueError, match="k1"):
        intrinsics_from_mapping({
            "fx": 1, "fy": 1, "cx": 0, "cy": 0,
            "color_distortion": {"k1": float("nan")},
        })


def test_intrinsics_reject_invalid_focal_length() -> None:
    with pytest.raises(ValueError, match="大于 0"):
        intrinsics_from_mapping({"fx": 0, "fy": 600, "cx": 1, "cy": 1})
