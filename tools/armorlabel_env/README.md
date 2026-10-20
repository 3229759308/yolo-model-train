# ArmorLabel 独立运行环境

该目录封装 `tools/bit-auto-label` 的环境和启动入口，不修改子模块源码。

## 启动

在 `/workspace` 执行：

```bash
bash tools/armorlabel_env/start.sh
```

这是 PySide6 桌面程序，不是网页。窗口显示到 Docker 的 `DISPLAY` 对应桌面。当前容器继承了有效的 `DISPLAY`，已实际验证 X11 连接成功。
也可明确只使用人工标注：

```bash
bash tools/armorlabel_env/start.sh --manual
```

环境在 `tools/bit-auto-label/.venv/`，容器全局 YOLO/PyTorch 环境独立保留。

## 已安装配置

| 组件 | 实际版本 |
|---|---|
| Python | 3.13.15 |
| PyTorch / CUDA | 2.10.0+cu128 / 12.8 |
| torchvision | 0.25.0+cu128 |
| PySide6 | 6.11.1 |
| NumPy | 2.3.5 |
| OpenCV | 4.13.0.92 |
| SAM3 | 0.1.0（本地子模块可编辑安装） |
| Ultralytics | 8.4.21 |
| ONNX Runtime GPU | 1.24.4 |
| huggingface-hub | 1.6.0 |
| click | 8.5.0（补齐 hf 命令依赖） |

标注工具固定提交：`cba76fb33f43f7fa4f68c56e5c550fc1be3bb3d2`。
SAM3 固定提交：`8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`。
上游完整业务依赖见 `../bit-auto-label/requirements.txt`，Linux 图形库与中文字体安装列表见 `system-deps.sh`。
以上是本次实际安装版本；重建脚本选择 Python 3.13 系列，未锁定 Python 补丁版本及全部传递依赖。

已验证：RTX 4060 Ti 的 CUDA 矩阵运算、SAM3 模块导入、ONNX Runtime 导入，以及真实 X11 窗口启动、事件循环与正常退出。

## SAM3 自动辅助标注

人工标注无需 SAM3 权重；缺少权重时，启动器跳过模型加载，仍可打开标注界面。
SAM3 Python 包及其依赖随环境安装。本次已从用户指定的公开镜像下载权重，保存为：

```text
tools/bit-auto-label/models/sam3.pt
```

官方来源：https://huggingface.co/facebook/sam3 （需要申请访问并登录）。
本次下载来源：https://huggingface.co/cubert-gmbh/sam3 。
文件大小：3,450,062,241 字节。下载后已确认 SHA256 与镜像公布值一致：

```text
9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
```

重新下载镜像权重（先完成环境安装，在 `/workspace` 执行）：

```bash
HF_HOME=/workspace/.cache/huggingface \
  tools/bit-auto-label/.venv/bin/hf download cubert-gmbh/sam3 sam3.pt \
  --local-dir /workspace/tools/bit-auto-label/models
sha256sum tools/bit-auto-label/models/sam3.pt
```

已使用 `torch.load(weights_only=True, mmap=True)` 验证权重可读取；尚未验证完整 SAM3 模型加载和图片辅助标注推理。
重新启动时不传 `--manual` 即按原程序加载权重。

## 重建环境

当前容器已安装 Linux 图形库与中文字体。若重建 Docker 容器，先以 root 执行：

```bash
bash tools/armorlabel_env/system-deps.sh
```

再安装 Python 环境（需要 uv 和网络，下载数 GB 的 CUDA 依赖）：

```bash
bash tools/armorlabel_env/setup.sh
```

Python 下载与包缓存位于 `/workspace/.cache/`，不提交 Git。SAM3 按上游说明使用 `--no-deps` 安装，以保留项目锁定的 NumPy 2.x；这会保留 SAM3 包元数据中的 NumPy 版本声明差异。

启动器处理了 OpenCV 与 PySide6 的 Qt 插件路径冲突。检查启动和退出：

```bash
bash tools/armorlabel_env/start.sh --manual --smoke-test
```

## 显示连接

如果换容器后没有图形显示，需在启动容器时挂载 `/tmp/.X11-unix`，设置正确的 `DISPLAY`，并使用宿主机的 X11 认证。单独激活 Python 环境不能替代图形显示连接。

## Git 提交范围

只提交子模块 URL 与固定提交指针、环境辅助脚本和说明。不提交虚拟环境、Python 安装目录、包缓存、权重、视频或图片数据；本机安装的 apt 软件包也不进入 Git。重新克隆后需要按上述步骤重建环境和下载权重。
