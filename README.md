# YOLO 训练工作区

在当前 Docker 容器内管理 YOLO 训练代码及配置。

建议目录：

- `src/`：训练、验证和推理代码
- `configs/`：模型及数据集 YAML 配置，可提交到 Git
- `data/` 或 `datasets/`：本地数据集，不提交
- `weights/`：预训练权重，不提交
- `runs/`：训练结果，不提交

当前容器已安装 Ultralytics 和 PyTorch，并已检测到可用的 NVIDIA GPU。

## 装甲板自定义检测头训练

训练使用 YOLOv8n-Pose 骨干和自定义头，独立输出 4 种颜色、9 种类别、4 个点，固定输入 `1×3×480×640`、输出 `1×21×6300`。使用统一入口：

```bash
python src/armor.py models
python src/armor.py train --config configs/armor_pose/yolov8n/empty10.yaml
```

`empty10.yaml` 使用保留原空图 10% 的数据集；完整原数据配置为 [default.yaml](configs/armor_pose/yolov8n/default.yaml)。数据、网络、损失、训练和部署实现分在 `src/armor_pose/` 对应子包，配置按骨干放在 `configs/armor_pose/`。准备、空图抽样、验证、预测、导出、目录树和学习顺序见 [训练与代码说明](src/README.md)。

目前可训练模型为 `yolov8n-armor-pose`（别名 `yolov8n`）；YOLOv5 和 YOLO26 目录仅预留适配说明，尚未实现。旧 CLI 薄包装和旧 YAML 相对链接保留兼容。

## Git 使用

查看修改：`git status`

首次提交前，设置本仓库的提交身份：

```bash
git config user.name "你的名字"
git config user.email "你的邮箱"
git add README.md .gitignore
git commit -m "Initialize YOLO training workspace"
```

数据集、权重及训练输出已在 `.gitignore` 中排除。不要将密钥写入代码或配置文件。

## 视频抽帧

网页抽帧与命令行批量抽帧工具位于 `tools/frame_extractor/`。运行方式和参数见 [抽帧工具 README](tools/frame_extractor/README.md)。

## 装甲板标注工具

`tools/bit-auto-label` 是 Git 子模块，包含其自身的 SAM3 子模块。克隆后初始化：

```bash
git submodule update --init --recursive
```

环境版本、系统依赖、权重来源与校验值，以及重建步骤记录在 [标注环境 README](tools/armorlabel_env/README.md)。已配置环境后启动：

```bash
bash tools/armorlabel_env/start.sh
```

该工具使用独立 Python 环境，以 PySide6 桌面窗口运行。Git 仅保存源码引用、辅助脚本及配置说明；虚拟环境、下载缓存、模型权重和数据集均不提交。

## 自定义检测头标注工具

`tools/armor_labeler` 提供独立颜色、九类装甲板和四个原图像素关键点标注，支持手工编辑；默认加载 `weights/Infantry-v8n-fp16-20260726-D1.8w-B16.onnx`，输入 `1×3×480×640`、输出 `1×21×6300`：

```bash
bash tools/armor_labeler/start.sh --images /workspace/data/raw_image/test
```

默认标签写入图片目录旁的 `<目录名>_labels`，可用 `--labels` 指定。随附配置按真实模型的网络输入坐标还原到原图；预处理和输出坐标可在窗口内切换。配置中的模型路径相对配置文件解析，`--model` 可覆盖。工具默认复用 `bit-auto-label/.venv`；详细操作、模型证据、依赖与导出说明见 [工具 README](tools/armor_labeler/README.md)。
