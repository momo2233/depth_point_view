from __future__ import annotations

import hashlib
import io
import math
from functools import lru_cache
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from depth_viewer.loaders import (
    load_depth_bytes,
    load_intrinsics_bytes,
    load_rgb_bytes,
    make_intrinsics,
)
from depth_viewer.models import Intrinsics, PointCloud
from depth_viewer.ply import encode_binary_ply
from depth_viewer.pointcloud_io import load_ply_bytes, sample_point_cloud
from depth_viewer.processing import (
    COLORMAPS,
    depth_statistics,
    depth_to_meters,
    encode_png,
    project_depth_to_point_cloud,
    render_pseudocolor,
    suggest_depth_scale,
    valid_depth_mask,
    validate_intrinsics_size,
)


MAX_PREVIEW_POINTS = 50_000
IMAGE_PREVIEW_MAX_WIDTH = 1600
IMAGE_PREVIEW_MAX_HEIGHT = 1000
T = TypeVar("T")


@st.cache_data(show_spinner=False, max_entries=8)
def cached_load_intrinsics(data: bytes, filename: str):
    return load_intrinsics_bytes(data, filename)


def session_cached(name: str, token: tuple, factory: Callable[[], T]) -> T:
    entry = st.session_state.get(name)
    if entry is None or entry[0] != token:
        entry = (token, factory())
        st.session_state[name] = entry
    return entry[1]


def preview_stride(shape: tuple[int, ...]) -> int:
    height, width = shape[:2]
    return max(
        1,
        math.ceil(width / IMAGE_PREVIEW_MAX_WIDTH),
        math.ceil(height / IMAGE_PREVIEW_MAX_HEIGHT),
    )


def preview_pixels(image: np.ndarray) -> np.ndarray:
    step = preview_stride(image.shape)
    return np.ascontiguousarray(image[::step, ::step])


def build_depth_context(depth: np.ndarray, scale: float) -> tuple[np.ndarray, dict]:
    depth_m = depth_to_meters(depth, scale)
    return depth_m, depth_statistics(depth_m)


def cloud_css_colors(colors: np.ndarray) -> list[str]:
    return [f"rgb({red},{green},{blue})" for red, green, blue in colors.tolist()]


def short_token(data: bytes, suffix: str = "") -> str:
    digest = hashlib.sha256(data).hexdigest()[:12]
    return f"{digest}_{suffix}" if suffix else digest


def format_number(value: float) -> str:
    return f"{value:.15g}"


def parse_form_intrinsics(
    values: dict[str, str], file_intrinsics: Intrinsics | None
) -> tuple[Intrinsics | None, str | None]:
    stripped = {key: value.strip() for key, value in values.items()}
    if not any(stripped.values()):
        return None, None
    if not all(stripped.values()):
        return None, "请完整填写 fx、fy、cx、cy"
    try:
        numbers = {key: float(value) for key, value in stripped.items()}
        return (
            make_intrinsics(
                **numbers,
                width=file_intrinsics.width if file_intrinsics else None,
                height=file_intrinsics.height if file_intrinsics else None,
                depth_scale=file_intrinsics.depth_scale if file_intrinsics else None,
                distortion=file_intrinsics.distortion if file_intrinsics else None,
            ),
            None,
        )
    except ValueError as exc:
        return None, str(exc)


def build_cloud_figure(
    cloud: PointCloud, point_size: int, dark: bool, axis_unit: str = "m"
) -> go.Figure:
    background = "#07111f" if dark else "#ffffff"
    grid = "#30445f" if dark else "#d8dee8"
    font = "#e6edf6" if dark else "#172033"
    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=cloud.points[:, 0],
                y=cloud.points[:, 1],
                z=cloud.points[:, 2],
                mode="markers",
                marker={
                    "size": point_size,
                    "color": cloud_css_colors(cloud.colors),
                    "opacity": 1.0,
                },
                hovertemplate=(
                    f"X: %{{x:.4f}} {axis_unit}<br>Y: %{{y:.4f}} {axis_unit}"
                    f"<br>Z: %{{z:.4f}} {axis_unit}<extra></extra>"
                ),
            )
        ]
    )
    figure.update_layout(
        height=720,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor=background,
        font={"color": font},
        showlegend=False,
        scene={
            "bgcolor": background,
            "aspectmode": "data",
            "xaxis": {"title": f"X / {axis_unit}", "gridcolor": grid, "zerolinecolor": grid},
            "yaxis": {"title": f"Y / {axis_unit}", "gridcolor": grid, "zerolinecolor": grid},
            "zaxis": {"title": f"Z / {axis_unit}", "gridcolor": grid, "zerolinecolor": grid},
            "camera": {
                "eye": {"x": 0.0, "y": 0.0, "z": -1.7},
                "up": {"x": 0.0, "y": -1.0, "z": 0.0},
            },
        },
    )
    return figure


def render_uploaded_cloud(upload, input_generation: int) -> None:
    if upload is None:
        st.info("上传 PLY 点云文件后，可在这里直接进行 3D 交互查看。")
        return
    data = upload.getvalue()
    token = short_token(data, str(input_generation))
    try:
        bundle = session_cached(
            "_uploaded_cloud_bundle",
            (token, upload.name),
            lambda: load_ply_bytes(data, upload.name),
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    for warning in bundle.warnings:
        st.warning(warning)
    cloud = bundle.cloud
    controls = st.columns([1, 1, 1])
    with controls[0]:
        point_size = st.slider("点大小", 1, 6, 2, key=f"uploaded_size_{token}")
    with controls[1]:
        dark_background = st.selectbox(
            "背景", ["深色", "浅色"], key=f"uploaded_background_{token}"
        ) == "深色"
    with controls[2]:
        preview_limit = st.selectbox(
            "预览点数上限",
            [20_000, MAX_PREVIEW_POINTS, 100_000, 200_000],
            index=1,
            format_func=lambda count: f"{count // 10_000} 万",
            key=f"uploaded_limit_{token}",
        )
    preview = session_cached(
        "_uploaded_cloud_preview",
        (token, preview_limit),
        lambda: sample_point_cloud(cloud, preview_limit),
    )

    minimum = np.min(cloud.points, axis=0)
    maximum = np.max(cloud.points, axis=0)
    metrics = st.columns(3)
    metrics[0].metric("点数", f"{cloud.points.shape[0]:,}")
    metrics[1].metric("预览点", f"{preview.points.shape[0]:,}")
    metrics[2].metric(
        "空间范围",
        f"{np.linalg.norm(maximum - minimum):.4g}",
        help="点云包围盒对角线长度，单位沿用文件中的坐标单位。",
    )
    st.plotly_chart(
        build_cloud_figure(preview, point_size, dark_background, axis_unit="文件单位"),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": True},
        key=f"uploaded_plot_{token}",
    )
    st.caption(
        f"包围盒：X [{minimum[0]:.4g}, {maximum[0]:.4g}] · "
        f"Y [{minimum[1]:.4g}, {maximum[1]:.4g}] · "
        f"Z [{minimum[2]:.4g}, {maximum[2]:.4g}]。鼠标拖动旋转，滚轮缩放。"
    )


def render_intro() -> None:
    st.info(
        "请在左侧上传 depth 图开始 RGB-D 查看，或直接上传 PLY 点云文件进行 3D 查看。"
    )
    st.markdown(
        """
        **支持格式**

        - Depth：PNG、TIFF、NPY、NPZ（二维整型或浮点数组）
        - RGB：PNG、JPEG、TIFF、BMP、WebP
        - 内参：JSON、YAML，支持 `fx/fy/cx/cy`、`K` 或 `camera_matrix.data`
        - 点云：ASCII 或二进制 PLY，支持有颜色或无颜色顶点
        """
    )


def render_rgb_summary(rgb: np.ndarray) -> None:
    summary = st.columns(3)
    summary[0].metric("RGB 分辨率", f"{rgb.shape[1]} × {rgb.shape[0]}")
    summary[1].metric("数据类型", str(rgb.dtype))
    summary[2].metric("颜色通道", "RGB · 3 通道")


def reset_current_data() -> None:
    generation = int(st.session_state.get("input_generation", 0)) + 1
    st.cache_data.clear()
    for key in list(st.session_state):
        del st.session_state[key]
    st.session_state["input_generation"] = generation


def main() -> None:
    st.set_page_config(page_title="RGB-D 可视化工具", page_icon="◉", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
        [data-testid="stMetricValue"] {font-size: 1.35rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("RGB-D 图像与点云可视化工具")
    st.caption("所有文件仅在本机内存中处理，不会上传到外部服务。")

    with st.sidebar:
        st.header("数据输入")
        if "input_generation" not in st.session_state:
            st.session_state["input_generation"] = 0
        if st.button(
            "一键清理当前数据",
            type="primary",
            width="stretch",
            help="清空 RGB、depth、内参、页面设置和已生成的缓存文件。",
        ):
            reset_current_data()
            st.rerun()
        input_generation = st.session_state["input_generation"]
        depth_upload = st.file_uploader(
            "Depth 文件 *",
            type=["png", "tif", "tiff", "npy", "npz"],
            key=f"depth_upload_{input_generation}",
        )
        rgb_upload = st.file_uploader(
            "RGB 文件",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
            key=f"rgb_upload_{input_generation}",
        )
        intrinsics_upload = st.file_uploader(
            "内参文件",
            type=["json", "yaml", "yml"],
            key=f"intrinsics_upload_{input_generation}",
        )
        cloud_upload = st.file_uploader(
            "直接查看点云（PLY）",
            type=["ply"],
            key=f"cloud_upload_{input_generation}",
        )

    if depth_upload is None:
        if cloud_upload is None:
            render_intro()
        else:
            st.subheader("上传点云")
            render_uploaded_cloud(cloud_upload, input_generation)
        return

    depth_bytes = depth_upload.getvalue()
    depth_token = short_token(depth_bytes)
    try:
        depth_bundle = session_cached(
            "_depth_bundle",
            (depth_token, depth_upload.name),
            lambda: load_depth_bytes(depth_bytes, depth_upload.name),
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    with st.sidebar:
        for warning in depth_bundle.warnings:
            st.warning(warning)
        depth_keys = list(depth_bundle.arrays)
        if len(depth_keys) > 1:
            depth_key = st.selectbox("NPZ 数组", depth_keys, key=f"array_{depth_token}")
        else:
            depth_key = depth_keys[0]
        depth = depth_bundle.arrays[depth_key]
        st.caption(f"Depth：{depth.shape[1]} × {depth.shape[0]} · {depth.dtype}")

    rgb = None
    rgb_error = None
    rgb_token = "none"
    if rgb_upload is not None:
        try:
            rgb_bytes = rgb_upload.getvalue()
            rgb_token = short_token(rgb_bytes)
            rgb_bundle = session_cached(
                "_rgb_bundle",
                (rgb_token, rgb_upload.name),
                lambda: load_rgb_bytes(rgb_bytes, rgb_upload.name),
            )
            rgb = rgb_bundle.image
            with st.sidebar:
                for warning in rgb_bundle.warnings:
                    st.warning(warning)
                st.caption(f"RGB：{rgb.shape[1]} × {rgb.shape[0]} · uint8")
            if rgb.shape[:2] != depth.shape:
                rgb_error = (
                    f"RGB 尺寸 {rgb.shape[1]}×{rgb.shape[0]} 与 depth 尺寸 "
                    f"{depth.shape[1]}×{depth.shape[0]} 不一致，不能生成彩色点云。"
                )
        except ValueError as exc:
            rgb_error = str(exc)

    file_intrinsics = None
    intrinsics_file_error = None
    intrinsics_token = "manual"
    if intrinsics_upload is not None:
        intrinsics_data = intrinsics_upload.getvalue()
        intrinsics_token = short_token(intrinsics_data)
        try:
            file_intrinsics = cached_load_intrinsics(
                intrinsics_data, intrinsics_upload.name
            )
        except ValueError as exc:
            intrinsics_file_error = str(exc)

    with st.sidebar:
        st.divider()
        st.subheader("相机参数")
        if intrinsics_file_error:
            st.error(intrinsics_file_error)
        elif file_intrinsics:
            st.success("已读取内参文件，可在下方覆盖参数")

        defaults = file_intrinsics or Intrinsics(0.0, 0.0, 0.0, 0.0)
        form_values = {
            "fx": st.text_input(
                "fx", value=format_number(defaults.fx) if defaults.fx else "", key=f"fx_{intrinsics_token}"
            ),
            "fy": st.text_input(
                "fy", value=format_number(defaults.fy) if defaults.fy else "", key=f"fy_{intrinsics_token}"
            ),
            "cx": st.text_input(
                "cx", value=format_number(defaults.cx) if file_intrinsics else "", key=f"cx_{intrinsics_token}"
            ),
            "cy": st.text_input(
                "cy", value=format_number(defaults.cy) if file_intrinsics else "", key=f"cy_{intrinsics_token}"
            ),
        }
        intrinsics, intrinsics_error = parse_form_intrinsics(form_values, file_intrinsics)
        if intrinsics_error:
            st.error(intrinsics_error)

        default_scale = (
            file_intrinsics.depth_scale
            if file_intrinsics and file_intrinsics.depth_scale is not None
            else suggest_depth_scale(depth.dtype)
        )
        scale_token = f"{depth_token}_{depth_key}_{intrinsics_token}"
        depth_scale = st.number_input(
            "原始值 → 米倍率",
            min_value=0.000000000001,
            max_value=1_000_000.0,
            value=float(default_scale),
            step=0.000001,
            format="%.12f",
            key=f"scale_{scale_token}",
            help="整数 depth 默认建议 0.001（毫米），浮点 depth 默认建议 1.0（米）。",
        )
        st.caption(f"当前换算：1 原始单位 = {depth_scale:.12g} 米")

    tab_rgb, tab_depth, tab_cloud, tab_uploaded_cloud = st.tabs(
        ["RGB 图像", "Depth 伪彩图", "RGB-D 点云", "上传点云"],
        on_change="rerun",
        key=f"view_{input_generation}",
    )

    depth_m = None
    stats = None
    if tab_depth.open or tab_cloud.open:
        depth_m, stats = session_cached(
            "_depth_context",
            (depth_token, depth_key, float(depth_scale)),
            lambda: build_depth_context(depth, depth_scale),
        )

    with tab_rgb:
        if not tab_rgb.open:
            pass
        elif rgb_upload is None:
            st.info("尚未上传 RGB 图像。伪彩图可独立查看，彩色点云需要 RGB。")
        elif rgb_error:
            st.error(rgb_error)
            if rgb is not None:
                render_rgb_summary(rgb)
                st.image(preview_pixels(rgb), caption=rgb_upload.name, width="stretch")
        else:
            render_rgb_summary(rgb)
            rgb_preview = preview_pixels(rgb)
            st.image(rgb_preview, caption=rgb_upload.name, width="stretch")
            if rgb_preview.shape != rgb.shape:
                st.caption(
                    f"页面预览已缩小至 {rgb_preview.shape[1]} × {rgb_preview.shape[0]}，"
                    "不影响点云使用的原始 RGB 数据。"
                )

    with tab_depth:
        if not tab_depth.open:
            pass
        elif stats["valid"] == 0:
            st.error("depth 中没有大于 0 的有限有效值，无法生成伪彩图。")
        else:
            metrics = st.columns(5)
            metrics[0].metric("Depth 分辨率", f"{depth.shape[1]} × {depth.shape[0]}")
            metrics[1].metric("有效像素", f"{int(stats['valid']):,}")
            metrics[2].metric("无效像素", f"{int(stats['invalid']):,}")
            metrics[3].metric("最小深度", f"{float(stats['min']):.4g} m")
            metrics[4].metric("最大深度", f"{float(stats['max']):.4g} m")
            st.caption(f"Depth 数据类型：{depth.dtype} · 单通道")

            auto_min = float(stats["p02"])
            auto_max = float(stats["p98"])
            if auto_max <= auto_min:
                auto_max = auto_min + max(abs(auto_min) * 1e-6, 1e-6)
            range_token = f"{depth_token}_{depth_key}_{depth_scale:.12g}"
            controls = st.columns([1.2, 1, 1, 0.8])
            with controls[0]:
                colormap = st.selectbox("配色方案", list(COLORMAPS), key=f"cmap_{range_token}")
            with controls[1]:
                display_min = st.number_input(
                    "显示最小值 / m",
                    value=float(auto_min),
                    format="%.8g",
                    key=f"display_min_{range_token}",
                )
            with controls[2]:
                display_max = st.number_input(
                    "显示最大值 / m",
                    value=float(auto_max),
                    format="%.8g",
                    key=f"display_max_{range_token}",
                )
            with controls[3]:
                reverse = st.toggle("反转配色", key=f"reverse_{range_token}")

            if display_max <= display_min:
                st.error("显示最大值必须大于显示最小值。")
            else:
                full_preview = st.toggle(
                    "原始分辨率预览（较慢）",
                    value=False,
                    key=f"full_preview_{range_token}",
                    help="默认缩小页面预览以便快速查看；下载的 PNG 始终为原始分辨率。",
                )
                view_depth = depth_m if full_preview else preview_pixels(depth_m)
                pseudo_png = session_cached(
                    "_pseudo_preview",
                    (
                        range_token,
                        float(display_min),
                        float(display_max),
                        colormap,
                        reverse,
                        full_preview,
                    ),
                    lambda: encode_png(
                        render_pseudocolor(
                            view_depth, display_min, display_max, colormap, reverse
                        )
                    ),
                )
                st.image(
                    io.BytesIO(pseudo_png),
                    caption=(
                        f"预览 {view_depth.shape[1]} × {view_depth.shape[0]} · "
                        "可右键复制或另存；下载按钮保存原始分辨率 PNG"
                    ),
                    width="stretch",
                )

                @lru_cache(maxsize=1)
                def make_full_png() -> bytes:
                    return encode_png(
                        render_pseudocolor(
                            depth_m, display_min, display_max, colormap, reverse
                        )
                    )

                pseudo_name = f"{Path(depth_upload.name).stem}_pseudocolor.png"
                st.download_button(
                    "下载伪彩图 PNG",
                    data=make_full_png,
                    file_name=pseudo_name,
                    mime="image/png",
                    key=f"download_pseudocolor_{range_token}",
                    on_click="ignore",
                    width="stretch",
                )

    with tab_cloud:
        blocking_error = None
        if not tab_cloud.open:
            pass
        elif stats["valid"] == 0:
            blocking_error = "depth 中没有有效像素。"
        elif rgb_upload is None:
            blocking_error = "请上传与 depth 已配准且尺寸完全一致的 RGB 图像。"
        elif rgb_error:
            blocking_error = rgb_error
        elif intrinsics is None:
            blocking_error = "请上传内参文件或完整填写 fx、fy、cx、cy。"
        else:
            try:
                validate_intrinsics_size(intrinsics, depth.shape)
            except ValueError as exc:
                blocking_error = str(exc)

        if tab_cloud.open and blocking_error:
            st.info(blocking_error)
        elif tab_cloud.open:
            cloud_token = (
                f"{depth_token}_{depth_key}_{depth_scale:.12g}_"
                f"{rgb_token}_{intrinsics_token}"
            )
            cloud_controls = st.columns([1, 1, 0.8, 0.8])
            with cloud_controls[0]:
                cloud_min = st.number_input(
                    "点云最小深度 / m",
                    value=float(stats["min"]),
                    format="%.8g",
                    key=f"cloud_min_{cloud_token}",
                )
            with cloud_controls[1]:
                cloud_max = st.number_input(
                    "点云最大深度 / m",
                    value=float(stats["max"]),
                    format="%.8g",
                    key=f"cloud_max_{cloud_token}",
                )
            with cloud_controls[2]:
                point_size = st.slider(
                    "点大小", 1, 6, 2, key=f"point_size_{cloud_token}"
                )
            with cloud_controls[3]:
                dark_background = st.selectbox(
                    "背景", ["深色", "浅色"], key=f"background_{cloud_token}"
                ) == "深色"

            preview_limit = st.selectbox(
                "预览点数上限",
                [20_000, MAX_PREVIEW_POINTS, 100_000, 200_000],
                index=1,
                format_func=lambda count: f"{count // 10_000} 万",
                key=f"cloud_limit_{cloud_token}",
                help="只影响网页交互预览，PLY 下载仍包含过滤范围内全部有效点。",
            )

            has_distortion = (
                intrinsics.distortion is not None and not intrinsics.distortion.is_zero
            )
            correct_distortion = st.toggle(
                "畸变矫正（对比点云效果）",
                value=False,
                disabled=not has_distortion,
                key=f"undistort_{cloud_token}",
                help="按 OpenCV 径向-切向模型校正反投影射线；不会重复执行 RGB-depth 对齐或修改图像。",
            )
            if has_distortion:
                st.caption(
                    "使用内参文件中的 color_distortion：k1–k6、p1、p2；"
                    "按 OpenCV Brown–Conrady 模型解释。若图像已去畸变，请保持关闭。"
                )
                if intrinsics.distortion.model is not None:
                    st.warning(
                        f"文件中的 model={intrinsics.distortion.model} 尚无厂商模型定义；"
                        "矫正开关按 OpenCV 模型试算，请对比效果。"
                    )
            else:
                st.caption("内参文件未提供非零 color_distortion，畸变矫正不可用。")

            if cloud_max < cloud_min:
                st.error("点云最大深度不能小于最小深度。")
            else:
                export_count = int(
                    np.count_nonzero(valid_depth_mask(depth_m, cloud_min, cloud_max))
                )
                if export_count == 0:
                    st.warning("当前深度范围内没有有效点。")
                else:
                    with st.spinner("正在生成点云…"):
                        preview_cloud = session_cached(
                            "_rgbd_preview_cloud",
                            (
                                depth_token,
                                depth_key,
                                float(depth_scale),
                                rgb_token,
                                intrinsics,
                                float(cloud_min),
                                float(cloud_max),
                                preview_limit,
                                correct_distortion,
                            ),
                            lambda: project_depth_to_point_cloud(
                                depth_m,
                                rgb,
                                intrinsics,
                                min_depth=cloud_min,
                                max_depth=cloud_max,
                                max_points=preview_limit,
                                correct_distortion=correct_distortion,
                            ),
                        )
                    point_metrics = st.columns(3)
                    point_metrics[0].metric("原始有效点", f"{int(stats['valid']):,}")
                    point_metrics[1].metric(
                        "预览点", f"{preview_cloud.points.shape[0]:,}"
                    )
                    point_metrics[2].metric("导出点", f"{export_count:,}")
                    st.plotly_chart(
                        build_cloud_figure(preview_cloud, point_size, dark_background),
                        width="stretch",
                        config={"displaylogo": False, "scrollZoom": True},
                    )
                    st.caption("坐标系：X 向右、Y 向下、Z 向前；鼠标拖动旋转，滚轮缩放。")

                    @lru_cache(maxsize=1)
                    def make_full_ply() -> bytes:
                        export_cloud = project_depth_to_point_cloud(
                            depth_m,
                            rgb,
                            intrinsics,
                            min_depth=cloud_min,
                            max_depth=cloud_max,
                            max_points=None,
                            correct_distortion=correct_distortion,
                        )
                        return encode_binary_ply(export_cloud)

                    correction_suffix = "_undistorted" if correct_distortion else ""
                    ply_name = (
                        f"{Path(depth_upload.name).stem}_pointcloud{correction_suffix}.ply"
                    )
                    st.download_button(
                        "下载全量点云 PLY",
                        data=make_full_ply,
                        file_name=ply_name,
                        mime="application/octet-stream",
                        key=f"download_pointcloud_{cloud_token}_{correct_distortion}",
                        on_click="ignore",
                        width="stretch",
                    )
                    st.caption("首次点击下载时才生成全量 PLY；高分辨率数据可能需要等待数秒。")

    with tab_uploaded_cloud:
        if tab_uploaded_cloud.open:
            render_uploaded_cloud(cloud_upload, input_generation)


if __name__ == "__main__":
    main()
