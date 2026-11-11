# YOLO 训练工作区

装甲板检测模型的训练、标注与导出工作区，开发环境以 Docker 容器形式提供。

## Development

### Pre-requirements:

- x86-64 架构，NVIDIA 显卡（RTX 20 系列及更新），驱动 ≥ 580
- 任意 Linux 发行版，或 WSL2
- [VSCode](https://code.visualstudio.com/)，安装 [Dev Containers 扩展](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [安装 Docker 并 配置代理（部分国家或地区）](docs/zh-cn/docker_with_proxy.md)
- [安装 NVIDIA 驱动与 NVIDIA Container Toolkit](docs/zh-cn/nvidia_docker_setup.md)

### Step 1：获取镜像

开发容器使用本机的 `yolo-train:latest` 镜像。从维护者处拿到镜像文件后导入：

```bash
docker load -i yolo-train.tar.gz
```

也可以从镜像仓库拉取，或在克隆仓库后用 `Dockerfile` 自行构建，参见 [镜像构建与分发指南](docs/zh-cn/build_docker_image.md)。

### Step 2：克隆并打开仓库

克隆仓库，注意需要使用 `recurse-submodules` 以克隆子模块：

```bash
git clone --recurse-submodules https://github.com/3229759308/yolo-model-train.git
```

在 VSCode 中打开仓库：

```bash
code ./yolo-model-train
```

按 `Ctrl+Shift+P`，在弹出的菜单中选择 `Dev Containers: Reopen in Container`。

VSCode 将拉起一个 `Docker` 容器，容器中已配置好完整训练环境，之后所有工作将在容器内进行。若提示 `image yolo-train:latest not found`，说明 Step 1 尚未完成。

如果 `Dev Containers` 在启动时卡住很长一段时间，可以尝试 [这个解决方案](docs/zh-cn/fix_devcontainer_stuck.md)。

### Step 3：确认环境

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
| ONNX / ONNX Runtime（CPU） | 1.23.2 / 1.31.1 |
| PySide6（标注工具界面） | 6.12.0 |
| opencode | 2.2.5 |

终端为 `zsh` + `oh-my-zsh`（`af-magic` 主题），`.script/` 下的指令已加入 `PATH`，命令历史保存在 `.devcontainer/.zsh_history`，重建容器后仍保留。

如需把本机的个人配置（如 opencode 的模型配置与会话）挂进容器，编辑 `.devcontainer/docker-compose.local.yml`（首次打开容器时自动生成，不提交 Git，文件内有示例），然后 `Rebuild Container`。

运行单元测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -t .
```

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
runs/<实验名>/        ⑤ 训练过程文件：args.yaml、results.csv、last.pt、日志
results/<实验名>/     ⑥ 结果：best.pt、best.onnx、指标、对比评估、推理输出
```

`src/`、`configs/` 提交到 Git；`data/`、`models/`、`runs/`、`results/` 只保存在本地。实验名取自训练配置的 `name`，`runs/` 与 `results/` 下的同名目录属于同一次实验；`runs/<实验名>` 只用于续训和排查，确认结果后可以删除。

其余目录：`tools/` 为抽帧与标注工具，`tests/` 为单元测试，`docs/` 为环境配置文档，`.script/` 为容器内可直接调用的指令。

## 装甲板自定义检测头训练

训练支持 YOLOv8n-Pose、原版 YOLOv5n 骨干和共享自定义头，独立输出 4 种颜色、9 种类别、4 个点，固定输入 `1×3×480×640`、输出 `1×21×6300`。使用统一入口：

```bash
python src/armor.py models
python src/armor.py train --config configs/armor_pose/yolov8n/empty10.yaml
```

`empty10.yaml` 使用保留原空图 10% 的数据集；完整原数据配置为 [default.yaml](configs/armor_pose/yolov8n/default.yaml)。数据、网络、损失、训练和部署实现分在 `src/armor_pose/` 对应子包，配置按骨干放在 `configs/armor_pose/`。训练过程文件写入 `runs/<name>/`，结果写入 `results/<name>/`，`name` 取自配置；同名实验已存在时拒绝覆盖，可用 `--name` 另起实验名。准备、空图抽样、验证、预测、导出、目录树和学习顺序见 [训练与代码说明](src/README.md)。

目前可训练模型为 `yolov8n-armor-pose` / `yolov8n` 和原版 v7.0 的 `yolov5n-armor-pose` / `yolov5n`；YOLO26 尚未实现。两个已实现模型共享独立颜色、类别和四点头。

YOLOv5n 对照训练：

```bash
python src/armor.py train --config configs/armor_pose/yolov5n/empty10.yaml
```

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

`tools/armor_labeler` 提供独立颜色、九类装甲板和四个原图像素关键点标注，支持手工编辑；默认加载 `models/teacher/Infantry-v8n-fp16-20260726-D1.8w-B16.onnx`，输入 `1×3×480×640`、输出 `1×21×6300`：

```bash
bash tools/armor_labeler/start.sh --images /workspace/data/raw_image/test
```

默认标签写入图片目录旁的 `<目录名>_labels`，可用 `--labels` 指定。随附配置按真实模型的网络输入坐标还原到原图；预处理和输出坐标可在窗口内切换。配置中的模型路径相对配置文件解析，`--model` 可覆盖。工具默认复用 `bit-auto-label/.venv`；详细操作、模型证据、依赖与导出说明见 [工具 README](tools/armor_labeler/README.md)。
