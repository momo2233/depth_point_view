# RGB-D 图像与点云可视化工具

一个完全在本机运行的 Streamlit 小工具，用于查看 RGB 图像、生成可调配色的 depth 伪彩图，以及通过相机内参生成可交互的彩色点云。

## 功能

- Depth：支持 PNG、TIFF、NPY、NPZ，保留整数和浮点深度精度。
- RGB：支持 PNG、JPEG、TIFF、BMP、WebP。
- 伪彩图：2%–98% 自动显示范围、手动范围、9 种配色、反转、右键复制和原始分辨率 PNG 下载。
- 内参：支持手工输入和 JSON/YAML 文件。
- 点云：浏览器中旋转、缩放、平移；预览默认最多 5 万点（可选 2 万、10 万或 20 万）；可切换畸变矫正对比几何效果。
- 直接点云查看：上传 ASCII 或二进制 PLY，无需 RGB、depth 或相机内参。
- 导出：当前过滤范围内的全量彩色点云，binary little-endian PLY 格式。

所有文件都只在运行工具的计算机内存中处理，不会发送到外部服务。

## 安装与启动

需要 Python 3.10 或更高版本。

已配置 `torch310` 环境时，直接运行：

```bash
conda activate torch310
python -m streamlit run main.py
```

若要新建独立环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

建议在激活环境后直接启动，不要通过 `conda run` 包裹常驻的 Streamlit 进程，以便终端的 `Ctrl+C` 更及时地传递给服务。正在生成大文件时，退出仍可能需要短暂等待。

浏览器通常会自动打开；否则访问终端显示的本地地址，一般为 `http://localhost:8501`。

运行测试：

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

## 使用流程

1. 在左侧上传 depth 文件。
2. 在“Depth 伪彩图”页选择配色、反转和显示范围。默认展示缩小的快速预览，可右键复制/另存；如需右键复制原始分辨率图像，请打开“原始分辨率预览（较慢）”。下载按钮始终保存原始分辨率的无损 PNG。
3. 上传与 depth **已经配准且宽高完全一致**的 RGB 图像。
4. 上传内参文件，或填写 `fx`、`fy`、`cx`、`cy`。
5. 确认“原始值 → 米倍率”。整型 depth 默认建议 `0.001`，浮点 depth 默认建议 `1.0`。
6. 在“RGB-D 点云”页调整过滤范围、点大小和背景；若内参文件带 `color_distortion`，可开关“畸变矫正”对比效果，再下载对应状态的全量 PLY。
7. 查看完毕后，点击侧栏顶部的“一键清理当前数据”，即可清空上传文件、页面设置和生成缓存并查看下一组数据。

也可以只上传一个 PLY 文件，在没有 RGB-D 数据时直接进入交互式点云查看。PLY 顶点必须包含 `x/y/z`，颜色支持 `red/green/blue` 或 `r/g/b`；无颜色点云会使用默认蓝色。

## Depth 输入约定

- 数组必须为 `H×W`，也接受可压缩为 `H×W` 的 `H×W×1`。
- 支持整型和浮点型；`NaN`、无穷值及小于等于 0 的值作为无效深度。
- NPZ 中有多个可用二维数组时，页面会提供数组选择器。
- 多页 TIFF 只读取第一页并显示提示。
- 不支持有损 JPEG depth、RAW/BIN、EXR、视差图或带多个通道的 depth。

## 内参格式

推荐格式见 [`examples/intrinsics.yaml`](examples/intrinsics.yaml)：

```yaml
width: 640
height: 480
fx: 615.0
fy: 615.0
cx: 319.5
cy: 239.5
depth_scale: 0.001
```

也支持以下矩阵形式：

```json
{
  "width": 640,
  "height": 480,
  "K": [615.0, 0, 319.5, 0, 615.0, 239.5, 0, 0, 1],
  "depth_scale": 0.001
}
```

还可使用 `intrinsics` 或 `color` 嵌套对象、二维 `matrix`，以及 OpenCV 风格的 `camera_matrix.data`。对已对齐到彩色像素网格的 RGB-D 数据，可以在“RGB-D 点云”页打开“畸变矫正”对比结果；工具读取 `color_distortion` 的 `k1`–`k6`、`p1`、`p2`，按 OpenCV Brown–Conrady 径向/切向模型校正每个像素的反投影射线。该开关默认关闭，不会对 RGB/depth 图像重采样，也不会重复执行 D2C 对齐；预览和导出的 PLY 使用同一开关状态。`model` 字段若有值，界面会提醒该厂商编号尚未验证，仍按上述模型试算。如果图像已去畸变，不应再次开启矫正。如果文件声明了 `width`/`height` 且与 depth 不一致，工具会阻止生成点云。

## 坐标与导出

点云采用针孔相机模型：

```text
z = depth_m
x = (u - cx) * z / fx
y = (v - cy) * z / fy
```

坐标系为 X 向右、Y 向下、Z 向前。PLY 顶点包含 `float32 x/y/z` 和 `uint8 red/green/blue`。网页预览可能降采样，但 PLY 始终包含当前深度过滤范围内的全部有效点。

页面只处理当前打开的页签。伪彩图和点云的预览采用限尺寸或限点数策略；完整分辨率 PNG 和全量 PLY 在首次点击下载时生成，高分辨率文件可能需等待数秒。降低“预览点数上限”可以改善浏览器中的旋转流畅度，不会改变导出结果。
