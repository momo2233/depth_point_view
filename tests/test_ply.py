from __future__ import annotations

import numpy as np

from depth_viewer.models import PointCloud
from depth_viewer.ply import PLY_VERTEX_DTYPE, encode_binary_ply


def test_binary_ply_header_and_payload() -> None:
    cloud = PointCloud(
        points=np.array([[1.25, -2.5, 3.75], [4.0, 5.0, 6.0]], dtype=np.float32),
        colors=np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
    )

    encoded = encode_binary_ply(cloud)
    header, payload = encoded.split(b"end_header\n", maxsplit=1)

    assert b"format binary_little_endian 1.0" in header
    assert b"element vertex 2" in header
    vertices = np.frombuffer(payload, dtype=PLY_VERTEX_DTYPE)
    np.testing.assert_allclose(vertices["x"], [1.25, 4.0])
    np.testing.assert_allclose(vertices["y"], [-2.5, 5.0])
    np.testing.assert_allclose(vertices["z"], [3.75, 6.0])
    np.testing.assert_array_equal(vertices["red"], [10, 40])
    np.testing.assert_array_equal(vertices["green"], [20, 50])
    np.testing.assert_array_equal(vertices["blue"], [30, 60])


def test_empty_cloud_still_produces_valid_ply() -> None:
    cloud = PointCloud(
        points=np.empty((0, 3), dtype=np.float32),
        colors=np.empty((0, 3), dtype=np.uint8),
    )
    encoded = encode_binary_ply(cloud)
    assert b"element vertex 0\n" in encoded
    assert encoded.endswith(b"end_header\n")
