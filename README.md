# YOLO 训练工作区

在当前 Docker 容器内管理 YOLO 训练代码及配置。

建议目录：

- `src/`：训练、验证和推理代码
- `configs/`：模型及数据集 YAML 配置，可提交到 Git
- `data/` 或 `datasets/`：本地数据集，不提交
- `weights/`：预训练权重，不提交
- `runs/`：训练结果，不提交

当前容器已安装 Ultralytics 和 PyTorch，并已检测到可用的 NVIDIA GPU。

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

## 最小训练例程

在工作区运行：

```bash
python src/demo_train.py
```

首次运行会下载官方 COCO8 数据集及 YOLO11n 预训练权重。例程自动选择 GPU（无 GPU 时使用 CPU），训练 1 轮后保存验证集预测图片。

- 参数：`configs/demo.yaml`
- 数据集：`datasets/coco8/`
- 预训练权重：`weights/yolo11n.pt`
- 训练结果：`runs/demo_train/`，包括 `weights/best.pt`、`results.csv` 和图表
- 预测图片：`runs/demo_predict/`

重复运行会自动建立带数字后缀的结果目录，具体路径以终端输出为准。
这只是验证训练流程的例程，1 轮和极小数据集的指标不能代表正式模型效果。

数据集说明：https://docs.ultralytics.com/datasets/detect/coco8/

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
