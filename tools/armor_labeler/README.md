# 自定义装甲板图片标注工具

参考 `../bit-auto-label` 的四关键点工作流，使用独立的 PySide6 桌面窗口进行人工标注和 ONNX 预标注。颜色分支与装甲板类别分开保存，适配 YOLOv8n-Pose 骨干加自定义检测头。

已接入工作区真实模型 `weights/Infantry-v8n-fp16-20260726-D1.8w-B16.onnx`，随附配置默认加载它。该模型的实际输出为**网络输入坐标**，工具会还原成原图像素后供编辑和保存；这与最初描述的“模型直接输出原图坐标”不同，详见下方实测说明。

## 启动

```bash
bash tools/armor_labeler/start.sh --images /workspace/data/raw_image/test
```

不传 `--images` 时可在窗口内打开目录。`model_config.json` 中的 `model_path` 默认为 `../../weights/Infantry-v8n-fp16-20260726-D1.8w-B16.onnx`，相对配置文件所在目录解析。也可指定标签目录及另一份自定义 ONNX：

```bash
bash tools/armor_labeler/start.sh \
  --images /workspace/data/raw_image/test \
  --labels /workspace/data/armor_annotations \
  --model /path/to/v8n-20260726.onnx
```

`--config /path/to/config.json` 可加载自定义配置，`--model` 覆盖其中的默认模型路径。窗口右侧提供预处理方式和模型输出坐标的下拉选项，无需编辑 JSON；“配置”窗口还能调整类别名称、关键点名称、分支分数等参数。更换模型时请同时检查这些约定。

未指定 `--labels` 时，默认使用图片目录旁的 `<图片目录名>_labels`，例如 `/data/images` 对应 `/data/images_labels`。编辑后约 900 ms 自动保存，切换图片也会保存；无效或损坏的已有 JSON 会显示错误并阻止覆盖，请先修复该文件。

启动脚本优先使用环境变量 `ARMOR_LABELER_PYTHON` 指定的 Python，否则使用 `../bit-auto-label/.venv/bin/python`，最后使用 `python3`。新环境安装：

```bash
python3 -m venv /workspace/.venv-armor-labeler
/workspace/.venv-armor-labeler/bin/python -m pip install -r tools/armor_labeler/requirements.txt
ARMOR_LABELER_PYTHON=/workspace/.venv-armor-labeler/bin/python bash tools/armor_labeler/start.sh
```

桌面使用需要可用的 X11 或 Wayland 显示会话；容器或远程终端需带有正确的显示会话环境，例如 `DISPLAY`。无显示环境可运行文末的 offscreen 检查。

手动标注无需 ONNX Runtime。要使用 ONNX 预标注，在所选环境安装 `onnxruntime`；可在满足对应系统依赖时自行使用 `onnxruntime-gpu` 替代。

## 标注与配置

打开图片目录后，按 `N` 并依点序点击四点创建装甲板，无需预先画 bbox。右键撤回刚放的点，`Esc` 取消新建。在右侧选择独立的颜色和装甲类别；左键拖动点柄修改形状，拖动四边形内部移动整个目标，坐标表可输入小数像素。也可复制或删除选中目标。

默认开启“允许点位超出图片”，用于截断目标。可关闭该选项限制拖动范围。原生 JSON 与自定义 TXT 保留图外点；YOLO 导出拒绝越界点和退化 bbox，需先调整标注。

| 操作 | 快捷键 / 手势 |
| --- | --- |
| 打开图片目录 / 保存 | `Ctrl+O` / `Ctrl+S` |
| 上一张 / 下一张 | `A` / `D` |
| 新建四点 / 适应画布 | `N` / `F` |
| 缩放 / 平移 | 滚轮 / 中键或空格加左键拖动 |
| 微调选中点或整个目标 | 方向键，按住 `Shift` 每次 10 px |
| 撤销 / 重做 | `Ctrl+Z` / `Ctrl+Shift+Z` |
| 删除目标 / 当前图推理 | `Delete` / `I` |
| 查看帮助 | `F1` |

“起点轮换”和“反转点序”用于修正四点编号，工具不会按几何位置自动重排。缩放画布不会改变标注坐标。左侧支持文件名搜索，以及未保存、待复核、已复核筛选；确认无目标的图片也可勾选“当前图片已复核”。

当前图推理会追加候选供人工编辑；目录批量推理只处理**没有 JSON 标签**的图片，跳过所有已有标签，包括空标签和损坏标签，保护人工结果。自动结果始终标记为待复核。推理可停止；推理进行时关闭窗口会先请求停止，请在推理结束后再次关闭。

随附模型配置参照现有样例使用 `LT、LB、RB、RT`（左上、左下、右下、右上）。程序通用默认值 `default_config()` 仍使用 `P0、P1、P2、P3` 占位，便于适配其他点序。工具保留人工点序及模型输出顺序，请使标注、训练和部署三处一致。

### 类别与颜色定义

以下为用户提供的类别与颜色定义。类别 ID 对应输出通道 `4～12`，即 `输出通道 = Class ID + 4`；颜色 ID 对应输出通道 `0～3`。

| Class ID | 类别 | 备注 |
| --- | --- | --- |
| 0 | 哨兵 | |
| 1 | 英雄 | |
| 2 | 工程 | |
| 3 | 3号步兵 | |
| 4 | 4号步兵 | |
| 5 | 5号步兵 | v8n-0726 网络不再识别 |
| 6 | 前哨站 | |
| 7 | 基地（底部） | |
| 8 | 基地（顶部） | |

| Color ID | 颜色 | 备注 |
| --- | --- | --- |
| 0 | 蓝色 | |
| 1 | 红色 | |
| 2 | 白色 | |
| 3 | 紫色 | v8n-0726 网络不再识别 |

Class ID `5` 和 Color ID `3` 保留原编号，其余 ID 不压缩或重排，输出仍按 `4 个颜色分数 + 9 个类别分数 + 8 个关键点坐标` 解析。

界面显示名称由 `model_config.json` 的 `class_names` 和 `color_names` 按 ID 顺序确定。当前类别显示仍使用 `class_0` 至 `class_8` 占位，可按上表修改名称；重命名不改变类别 ID。

## 自定义头约定

- 真实模型输入：`float32[1,3,480,640]`。
- 真实模型输出：`float32[1,21,6300]`；解码器也兼容候选优先的 `1×6300×21` 排列。
- 文件名中的 `fp16` 表示图内半精度计算，实际输入和输出接口仍为 float32。
- 通道 `0～3`：蓝、红、白、紫分支分数。
- 通道 `4～12`：九种装甲类别分支分数。
- 通道 `13～20`：按配置点序排列的四组 `(x,y)`。

真实 ONNX 的关键点末端计算为 `(raw * 2 + grid) * stride`，得到网络输入空间坐标，没有原图尺寸或缩放逆变换。以 `1440×1080` 样例为例，原始输出的左上点约为 `(280.2,186.0)`；按 `2.25` 倍还原约为 `(630.5,418.5)`，与人工左上点约 `(631.5,412.2)` 接近。直接把网络输出当作原图像素会导致位置和尺寸明显偏小。

因此，随附 `model_config.json` 对此模型使用 `output_coordinates: "input"`；推理层按当前预处理将四点还原到原图。stretch 分别乘以原图与输入的宽高比；letterbox 先扣除实际整数 padding，再按实际 resize 后的宽高比例还原。允许图外点，不会在逆映射时裁剪。

对于确实直接输出原图像素的其他自定义头，可在“模型输出坐标”中选择原图，对应 `output_coordinates: "original"`，此时不执行逆变换。程序通用 `default_config()` 保持该值，以兼容最初约定。无论选哪种模型坐标，**画布、原生 JSON 和自定义 TXT 始终使用原图像素**。

默认预处理为直接 resize 至 `640×480`、BGR 转 RGB、除以 255，输出 float32 NCHW。可切换 letterbox、保留 BGR、关闭归一化；模型输入为 float16 时推理层自动转换。配置中的 `scores_are_logits` 可对分支分数应用 sigmoid，默认关闭。

解码分别选择颜色和类别分支得分最高的类别，默认置信度为两个分支分数之积，也可选择最小值或单独一个分支。程序通用默认置信度阈值为 0.25、IoU 阈值为 0.45；当前随附配置分别为 0.05 和 0.1，可在窗口中调整并自动保存。通过四点外接矩形执行 NMS，默认仅抑制同颜色且同类别的重叠候选。模型没有单独的 bbox 或 objectness 通道。

## 标签与导出

原生 JSON 保存原图尺寸、颜色 ID、装甲类别 ID、四点像素坐标及标注来源，用于无损继续编辑。标签目录保留图片相对目录与完整后缀，例如 `images/sub/a.jpg` 保存到 `labels/sub/a.jpg.json`，使 `sample.jpg` 与 `sample.png` 不会覆盖彼此。保存采用原子替换，第二次保存起，上一版本留在同名 `.json.bak`。

窗口导出默认仅包含已复核图片，可取消该筛选。每次导出建立新的 `armor_export_<时间>` 目录，标签放在 `labels/`，保留相对目录与图片后缀，例如 `sub/a.jpg.txt`。`manifest.json` 记录源图片根目录、图片和标签映射、点名及类别名；**图片不会复制**。任何坏标签或格式验证失败都会中止本次导出。

自定义 TXT 每个目标一行，共十个值：

```text
color_id class_id x0 y0 x1 y1 x2 y2 x3 y3
```

其中 `(x,y)` 为原图像素坐标。这里保存监督标签，不保存推理候选的分支分数。

标准 YOLO-Pose TXT 每行十七个值：

```text
combined_class cx cy width height x0 y0 2 x1 y1 2 x2 y2 2 x3 y3 2
```

`combined_class = color_id * 9 + class_id`，共 36 个组合类别；bbox 取四点外接矩形，坐标按图片宽高归一化，并附带 `classes.txt`。

导出的行内容遵循 YOLO-Pose，但文件名仍保留 `a.jpg.txt` 以避免同名不同后缀碰撞。标准 Ultralytics 数据集通常需要与图片 basename 对应的 `a.txt`，请按 `manifest.json` 映射整理图片及标签，并处理 `a.jpg` / `a.png` 同名情况、划分数据集及编写数据 YAML，再进行训练。自定义多分支训练应读取原生 JSON 或自定义 TXT。

## 检查

```bash
cd tools/armor_labeler
QT_QPA_PLATFORM=offscreen ../bit-auto-label/.venv/bin/python -m unittest discover -v -p 'test_*.py'
QT_QPA_PLATFORM=offscreen bash start.sh --smoke-test
```

测试发现命令覆盖核心、GUI 和 ONNX 适配；ONNX 测试临时生成合成模型，需要 `onnx` 和 `onnxruntime`，缺少时会跳过。用户提供的实际模型已另行完成图片推理及界面保存验证。无显示环境下的 smoke test 验证窗口初始化，不会代替真实桌面的鼠标交互验收。
