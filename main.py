from __future__ import annotations

import hashlib
import io
from pathlib import Path

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
    suggest_display_range,
    valid_depth_mask,
    validate_intrinsics_size,
)


MAX_PREVIEW_POINTS = 300_000


@st.cache_data(show_spinner=False, max_entries=8)
def cached_load_depth(data: bytes, filename: str):
    return load_depth_bytes(data, filename)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_load_rgb(data: bytes, filename: str):
    return load_rgb_bytes(data, filename)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_load_intrinsics(data: bytes, filename: str):
    return load_intrinsics_bytes(data, filename)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_load_uploaded_cloud(data: bytes, filename: str):
    return load_ply_bytes(data, filename)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_pseudocolor(
    depth_m: np.ndarray, vmin: float, vmax: float, colormap: str, reverse: bool
) -> bytes:
    return encode_png(render_pseudocolor(depth_m, vmin, vmax, colormap, reverse))


@st.cache_data(show_spinner=False, max_entries=8)
def cached_cloud(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    intrinsics: Intrinsics,
    min_depth: float,
    max_depth: float,
    max_points: int | None,
) -> PointCloud:
    return project_depth_to_point_cloud(
        depth_m,
        rgb,
        intrinsics,
        min_depth=min_depth,
        max_depth=max_depth,
        max_points=max_points,
    )


@st.cache_data(show_spinner=False, max_entries=8)
def cached_ply(cloud: PointCloud) -> bytes:
    return encode_binary_ply(cloud)


@st.cache_data(show_spinner=False, max_entries=8)
def cloud_css_colors(colors: np.ndarray) -> list[str]:
    return [f"rgb({red},{green},{blue})" for red, green, blue in colors.tolist()]


def short_token(data: bytes, suffix: str = "") -> str:
    digest = hashlib.sha256(data).hexdigest()[:12]
    return f"{digest}_{suffix}" if suffix else digest


def format_number(value: float) -> str:
    return f"{value:.8g}"


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
        bundle = cached_load_uploaded_cloud(data, upload.name)
    except ValueError as exc:
        st.error(str(exc))
        return

    for warning in bundle.warnings:
        st.warning(warning)
    cloud = bundle.cloud
    preview = sample_point_cloud(cloud, MAX_PREVIEW_POINTS)
    controls = st.columns([1, 1])
    with controls[0]:
        point_size = st.slider("点大小", 1, 6, 2, key=f"uploaded_size_{token}")
    with controls[1]:
        dark_background = st.selectbox(
            "背景", ["深色", "浅色"], key=f"uploaded_background_{token}"
        ) == "深色"

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
        depth_bundle = cached_load_depth(depth_bytes, depth_upload.name)
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
    if rgb_upload is not None:
        try:
            rgb_bundle = cached_load_rgb(rgb_upload.getvalue(), rgb_upload.name)
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

    depth_m = depth_to_meters(depth, depth_scale)
    stats = depth_statistics(depth_m)
    tab_rgb, tab_depth, tab_cloud, tab_uploaded_cloud = st.tabs(
        ["RGB 图像", "Depth 伪彩图", "RGB-D 点云", "上传点云"]
    )

    with tab_rgb:
        if rgb_upload is None:
            st.info("尚未上传 RGB 图像。伪彩图可独立查看，彩色点云需要 RGB。")
        elif rgb_error:
            st.error(rgb_error)
            if rgb is not None:
                render_rgb_summary(rgb)
                st.image(rgb, caption=rgb_upload.name, width="stretch")
        else:
            render_rgb_summary(rgb)
            st.image(rgb, caption=rgb_upload.name, width="stretch")

    with tab_depth:
        if stats["valid"] == 0:
            st.error("depth 中没有大于 0 的有限有效值，无法生成伪彩图。")
        else:
            metrics = st.columns(5)
            metrics[0].metric("Depth 分辨率", f"{depth.shape[1]} × {depth.shape[0]}")
            metrics[1].metric("有效像素", f"{int(stats['valid']):,}")
            metrics[2].metric("无效像素", f"{int(stats['invalid']):,}")
            metrics[3].metric("最小深度", f"{float(stats['min']):.4g} m")
            metrics[4].metric("最大深度", f"{float(stats['max']):.4g} m")
            st.caption(f"Depth 数据类型：{depth.dtype} · 单通道")

            auto_min, auto_max = suggest_display_range(depth_m)
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
                pseudo_png = cached_pseudocolor(
                    depth_m, display_min, display_max, colormap, reverse
                )
                st.image(
                    io.BytesIO(pseudo_png),
                    caption="可右键复制图片或另存为",
                    width="stretch",
                )
                pseudo_name = f"{Path(depth_upload.name).stem}_pseudocolor.png"
                st.download_button(
                    "下载伪彩图 PNG",
                    data=pseudo_png,
                    file_name=pseudo_name,
                    mime="image/png",
                    width="stretch",
                )

    with tab_cloud:
        blocking_error = None
        if stats["valid"] == 0:
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

        if blocking_error:
            st.info(blocking_error)
        else:
            cloud_token = f"{depth_token}_{depth_key}_{depth_scale:.12g}_{intrinsics_token}"
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
                        preview_cloud = cached_cloud(
                            depth_m,
                            rgb,
                            intrinsics,
                            cloud_min,
                            cloud_max,
                            MAX_PREVIEW_POINTS,
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

                    with st.spinner("正在准备全量 PLY…"):
                        export_cloud = cached_cloud(
                            depth_m,
                            rgb,
                            intrinsics,
                            cloud_min,
                            cloud_max,
                            None,
                        )
                        ply_data = cached_ply(export_cloud)
                    ply_name = f"{Path(depth_upload.name).stem}_pointcloud.ply"
                    st.download_button(
                        "下载全量点云 PLY",
                        data=ply_data,
                        file_name=ply_name,
                        mime="application/octet-stream",
                        width="stretch",
                    )

    with tab_uploaded_cloud:
        render_uploaded_cloud(cloud_upload, input_generation)


if __name__ == "__main__":
    main()
