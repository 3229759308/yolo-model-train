# YOLO 训练工作区

装甲板检测模型的抽帧、标注、训练与导出工作区，开发环境以 Docker 容器形式提供。

模型以 YOLOv8n-Pose 或原版 YOLOv5n v7.0 为骨干，接自定义检测头，分别输出 4 种颜色、9 种类别和装甲板的 4 个角点。部署用的 ONNX 固定输入 `1×3×480×640`、输出 `1×21×6300`，不含 NMS。

## 开发环境

### 前置要求

- x86-64 架构，NVIDIA 显卡（RTX 20 系列及更新），驱动 ≥ 580
- 任意 Linux 发行版，或 WSL2
- [VSCode](https://code.visualstudio.com/)，安装 [Dev Containers 扩展](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [安装 Docker 并 配置代理（部分国家或地区）](docs/zh-cn/docker_with_proxy.md)
- [安装 NVIDIA 驱动与 NVIDIA Container Toolkit](docs/zh-cn/nvidia_docker_setup.md)

### 第 1 步：获取镜像

开发容器使用本机的 `yolo-train:latest` 镜像。从维护者处拿到镜像文件后导入：

```bash
docker load -i yolo-train.tar.gz
```

也可以从镜像仓库拉取，或在克隆仓库后用 `Dockerfile` 自行构建，参见 [镜像构建与分发指南](docs/zh-cn/build_docker_image.md)。

### 第 2 步：克隆并打开仓库

克隆仓库：

```bash
git clone https://github.com/3229759308/yolo-model-train.git
```

在 VSCode 中打开仓库：

```bash
code ./yolo-model-train
```

按 `Ctrl+Shift+P`，在弹出的菜单中选择 `Dev Containers: Reopen in Container`。

VSCode 将拉起一个 `Docker` 容器，容器中已配置好完整训练环境，之后所有工作将在容器内进行。若提示 `image yolo-train:latest not found`，说明第 1 步尚未完成。

如果 `Dev Containers` 在启动时卡住很长一段时间，可以尝试 [这个解决方案](docs/zh-cn/fix_devcontainer_stuck.md)。

### 第 3 步：确认环境

容器创建完成后会自动运行一次环境检查，也可以在 VSCode 终端中随时输入：

```bash
check-env
```

输出以 `== Environment ready` 结尾即表示 GPU 与训练依赖均可用，否则参见 [NVIDIA 配置排查](docs/zh-cn/nvidia_docker_setup.md#容器内排查)。

容器内已包含：

| 组件 | 版本 |
| --- | --- |
| Ubuntu / Python | 24.04 / 3.12 |
| PyTorch / CUDA | 2.14.1 / 13.2 |
| torchvision | 0.29.1 |
| Ultralytics | 8.4.182 |
| OpenCV（headless） | 5.0 |
| ONNX / ONNX Runtime（GPU） | 1.23.2 / 1.31.1 |
| PySide6（标注工具界面） | 6.12.0 |
| opencode | 2.2.5 |

终端为 `zsh` + `oh-my-zsh`（`af-magic` 主题），`.script/` 下的指令已加入 `PATH`，命令历史保存在 `.devcontainer/.zsh_history`，重建容器后仍保留。

如需把本机的个人配置（如 opencode 的模型配置与会话）挂进容器，编辑 `.devcontainer/docker-compose.local.yml`（首次打开容器时自动生成，不提交 Git，文件内有示例），然后 `Rebuild Container`。

> 容器内用户为 `ubuntu`（可免密 `sudo`），其 UID 会自动映射为主机当前用户，容器内产生的文件在主机上直接归你所有。
>
> 从旧的 root 容器迁移过来时，先在**主机**仓库根目录执行一次 `sudo chown -R $USER: .`，否则旧的 root 文件在容器内无法写入。

## 目录约定

目录按数据流划分：

```text
data/                 ① 训练数据
├── videos/           录像，抽帧工具的默认输入
├── raw_image/        抽帧图片
├── raw_image_labels/ 标注工具保存的 JSON 标签
├── exports/          标注导出（manifest + TXT），prepare 的输入
├── datasets/         prepare / subsample 生成的训练数据集
└── reports/          数据检查报告
models/               ② 模型权重
├── pretrained/       预训练骨干
└── teacher/          标注工具使用的教师 ONNX
src/                  ③ 训练、验证、推理与导出代码
configs/              ④ 训练与数据集 YAML 配置
runs/<实验名>/        ⑤ 训练过程文件：args.yaml、results.csv、last.pt
results/<实验名>/     ⑥ 结果：best.pt、best.onnx、指标、对比评估、推理输出
```

`src/`、`configs/` 提交到 Git；`data/`、`models/`、`runs/`、`results/` 只保存在本地。实验名取自训练配置的 `name`，`runs/` 与 `results/` 下的同名目录属于同一次实验；`runs/<实验名>` 只用于续训和排查，确认结果后可以删除。

其余目录：`tools/` 为抽帧与标注工具，`tests/` 为单元测试，`docs/` 为环境配置文档，`.script/` 为容器内可直接调用的指令。

新环境需要自行放入 `models/` 中的权重：

| 文件 | 来源 |
| --- | --- |
| `pretrained/yolov8n-pose.pt` | [Ultralytics 官方](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt) |
| `pretrained/yolov5n.pt` | [YOLOv5 v7.0 官方](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt) |
| `teacher/Infantry-v8n-fp16-20260726-D1.8w-B16.onnx` | 向维护者获取 |

## 工作流程

以下命令都在容器内的仓库根目录执行，完整参数用 `--help` 查看；标签格式、网络结构和训练细节见 [训练与代码说明](src/README.md)。

### 1. 抽帧

把 AVI 录像放进 `data/videos/infantry/` 和 `data/videos/sentry/`，每 30 帧抽一张到 `data/raw_image/`：

```bash
python tools/frame_extractor/batch_extract.py
```

图片命名为 `<目录>_<视频名>_f<帧号>.jpg`。后续按名字中的视频划分训练集和验证集，同一段视频的帧不会同时出现在两边。需要逐个选择视频时可用网页工具，见 [抽帧工具 README](tools/frame_extractor/README.md)。

RMCS 录制的 `record.raw` 与 `record.csv` 用下面的命令导出 PNG，默认写入 `data/raw_image/<录制目录名>/`：

```bash
python tools/extract-raw-images <录制目录>
```

### 2. 标注与导出

```bash
bash tools/armor_labeler/start.sh --images data/raw_image
```

标签默认保存在 `data/raw_image_labels/`。工具默认用 `models/teacher/` 中的教师模型在 GPU 上预标注；用 `--model` 换成本项目导出的 ONNX 时，需在窗口右侧把预处理切换为 letterbox。

标注完成后点工具栏的「导出」，格式选「自定义头 TXT」，目标父目录选 `data/exports/`，工具会在其中新建 `armor_export_<时间>/`。导出默认只包含已复核的图片，用未复核的伪标签训练时要取消「仅导出已复核图片」。其余操作见 [标注工具 README](tools/armor_labeler/README.md)。

### 3. 准备数据集

```bash
python src/armor.py prepare --export data/exports/armor_export_<时间> --output data/datasets/<数据集名>
python src/armor.py subsample --source data/datasets/<数据集名> --output data/datasets/<数据集名>_empty10 --keep-empty 0.1
```

`prepare` 校验标签，按整段视频划出约 20% 作验证集，把图片和标签复制成新的数据集，输出目录已存在时拒绝覆盖。`subsample` 可选：保留全部有目标的图片，空图只保留原数量的 10%。现有数据集为 `armor_pose_custom` 及其抽样版 `armor_pose_custom_empty10`。

### 4. 训练

训练配置在 `configs/armor_pose/<骨干>/`，其中 `name` 是实验名，`data` 指向数据集：

| 配置 | 骨干 | 数据集 |
| --- | --- | --- |
| `yolov8n/default.yaml` | YOLOv8n-Pose | `armor_pose_custom` |
| `yolov8n/empty10.yaml` | YOLOv8n-Pose | `armor_pose_custom_empty10` |
| `yolov5n/empty10.yaml` | YOLOv5n v7.0 | `armor_pose_custom_empty10` |

```bash
python src/armor.py train --config configs/armor_pose/yolov8n/empty10.yaml --name yolov8n_empty10_v2
```

实验名已被使用时会拒绝覆盖，重新训练要用 `--name` 换一个实验名；换数据集时加 `--data data/datasets/<数据集名>/data.yaml`。训练中断后续训：

```bash
python src/armor.py train --resume runs/<实验名>/last.pt
```

### 5. 验证、导出与推理

```bash
python src/armor.py train --validate results/<实验名>/best.pt
python src/armor.py export --weights results/<实验名>/best.pt --verify
python src/armor.py predict --weights results/<实验名>/best.pt --source data/raw_image --save-images
```

`export` 生成 `results/<实验名>/best.onnx`，并用同一输入核对 ONNX 与 PyTorch 的输出；`predict` 把原图坐标的标签和可视化写入 `results/<实验名>/predictions/`。

## 测试与提交

```bash
PYTHONPATH=src python -m unittest discover -s tests -t .
```

提交说明遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/v1.0.0/)，例如 `feat(armor-pose): add ...`。
