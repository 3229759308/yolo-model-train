# 装甲板自定义 Pose 训练与部署

这套代码支持 **YOLOv8n-Pose 或原版 YOLOv5n v7.0 的骨干与特征融合网络，加共享自定义装甲板检测头**。颜色、装甲板类别和四个角点分别学习，保持原模型的 `4 + 9 + 8 = 21` 通道接口。统一入口是 `python src/armor.py train`，数据配置采用项目自己的格式，不能直接交给 `yolo pose train`。

当前训练使用工具导出的**伪标签监督**：读取已经保存的颜色、类别和四点坐标。没有读取教师 ONNX 的原始 logits，也没有加入教师特征或软输出蒸馏损失。验证集也是这些伪标签的一部分，指标反映模型与伪标签的一致性；实际识别质量还需要独立真实标注数据或实景检验。

## 1. 网络接口与坐标

部署 ONNX 固定输入为 `1×3×480×640`：RGB、浮点数、除以 255，使用灰值 114 的 letterbox 保持长宽比。训练时批量维度可以大于 1。

输出为 `1×21×6300`，不包含 NMS：

| 通道（从 0 开始） | 内容 |
| --- | --- |
| 0～3 | 颜色分支的 4 个 sigmoid 得分 |
| 4～12 | 类别分支的 9 个 sigmoid 得分 |
| 13～20 | `LT、LB、RB、RT` 四点依次排列的 `(x,y)` |

6300 个候选来自步长为 8、16、32 的三层特征图：`60×80 + 30×40 + 15×20`。四点顺序固定为**左上、左下、右下、右上**。

**网络输出的 XY 是 640×480 输入图的像素坐标。原始标签的 XY 是原图像素坐标。** 数据加载器会按照实际 resize 后的整数尺寸转换标签；推理时按同一组参数还原：

```text
gain = min(640 / original_w, 480 / original_h)
resized_w = round(original_w × gain)
resized_h = round(original_h × gain)
scale_x = resized_w / original_w
scale_y = resized_h / original_h

original_x = (network_x - padding_left) / scale_x
original_y = (network_y - padding_top) / scale_y
```

`padding_left`、`padding_top` 都是实际使用的整数补边量。不要将本次导出的模型设置成“直接输出原图坐标”，也不要用直接拉伸预处理代替 letterbox。

## 2. 类别与颜色定义

类别和颜色是两个独立 ID，训练头保留完整的 9 类和 4 色，不转换成 36 个分类输出。

| Class ID | 类别 |
| --- | --- |
| 0 | 哨兵 |
| 1 | 英雄 |
| 2 | 工程 |
| 3 | 3号步兵 |
| 4 | 4号步兵 |
| 5 | 5号步兵；v8n-0726 教师不再识别，当前数据没有此类样本 |
| 6 | 前哨站 |
| 7 | 基地（底部） |
| 8 | 基地（顶部） |

| Color ID | 颜色 |
| --- | --- |
| 0 | 蓝色 |
| 1 | 红色 |
| 2 | 白色 |
| 3 | 紫色；v8n-0726 教师不再识别，当前数据没有此色样本 |

没有正样本的输出槽位仍然保留，但这批数据无法教会模型识别对应类别或颜色。英文训练名称和完整中文定义写在准备后数据集的 `class_mapping.json` 中。

## 3. 准备数据集

在 `/workspace` 下执行。训练使用当前全局 Python 环境中的 Torch、Ultralytics、OpenCV、NumPy、Pillow 和 PyYAML。骨干预训练权重从本地 `/workspace/weights/yolov8n-pose.pt` 加载，不需要下载教师模型。当前数据集已经准备完成，可直接进入训练；下面的 prepare 命令用于准备新导出，已有输出目录会拒绝覆盖。

```bash
cd /workspace
python src/armor.py prepare
```

默认读取：

```text
/workspace/data/raw_image_label_end/armor_export_20261023_071657_908993
```

默认输出：

```text
/workspace/datasets/armor_pose_custom
├── data.yaml
├── images/train/...
├── images/val/...
├── labels/train/...
├── labels/val/...
├── class_mapping.json
├── preparation_report.json
└── split_manifest.json
```

每个标签目标是一行 10 列，没有 bbox、归一化坐标或可见性列：

```text
color_id class_id x_LT y_LT x_LB y_LB x_RB y_RB x_RT y_RT
```

准备过程遵循以下规则：

- 使用 `manifest.json` 中的全部图片，不按 `reviewed` 筛选，不设置人工复核门槛。
- 空标签图片保留，输出对应空 TXT；训练时参与背景分支监督。
- 四点使用原图像素坐标。图外点保留原值，不裁剪、不置零、不丢弃目标。
- 验证 ID 范围、10 列结构、有限坐标、有效图片和非零面积四点外接框；遇到不合法数据明确报错。
- 原图和导出标签只读。图片复制到新数据集，标签写入新目录；不修改输入文件。
- 保留图片子目录，标签改成对应的 `stem.txt`。不同图片后缀产生同一标签名时直接拒绝，避免覆盖。
- 输出目录已存在时拒绝覆盖；全部验证和复制成功后才发布完整目录。需要重建时使用新的 `--output`。

按文件名 `<视频前缀>_f<帧号>` 将同一视频的所有帧放进同一个 split，防止同组相邻帧同时出现在训练和验证中。默认 `--val-ratio 0.2 --seed 42`，确定性选择整组视频，兼顾图片数和常见类别覆盖；实际比例受视频组大小限制。

已经完成的当前划分为 9319 张训练图片、2335 张验证图片；共 3949 个目标、8989 张空标签图片。保留了 177 个目标中的 286 个图外关键点。验证视频为 `infantry_bisai_2`、`infantry_bisai_7`、`sentry_bisai_2`。原本存在的 `blue_base_bottom`、`red_base_top`、`white_engineer` 没有分配到验证集，这是整视频隔离和稀有类分布的限制，不能据此评估这些组合的验证性能。

自定义输入、输出或指定验证视频：

```bash
python src/armor.py prepare \
  --export /workspace/data/raw_image_label_end/armor_export_20261023_071657_908993 \
  --output /workspace/datasets/armor_pose_custom_v2 \
  --val-ratio 0.2 --seed 42

python src/armor.py prepare \
  --output /workspace/datasets/armor_pose_custom_explicit \
  --val-groups infantry_bisai_2 infantry_bisai_7 sentry_bisai_2
```

统计报告记录每个视频、split、类别和颜色的数量，以及稀有类别和缺失组合。报告中的 `color_id * 9 + class_id` 只用于联合统计和 AP 匹配；标签仍保留独立颜色与类别。

## 4. 正式训练

训练配置按骨干分类存放。当前空图减少后的训练使用 [`configs/armor_pose/yolov8n/empty10.yaml`](../configs/armor_pose/yolov8n/empty10.yaml)：

```bash
python src/armor.py train --config configs/armor_pose/yolov8n/empty10.yaml
```

完整原数据集的默认配置是 [`configs/armor_pose/yolov8n/default.yaml`](../configs/armor_pose/yolov8n/default.yaml)，省略 `--config` 时使用它。数据描述位于准备后目录的 `data.yaml`；[`configs/armor_pose/datasets/armor.yaml`](../configs/armor_pose/datasets/armor.yaml) 是相同自定义数据结构的静态配置。两个训练配置都指定 `model: yolov8n-armor-pose`。

显式指定输出目录和训练规模：

```bash
python src/armor.py train \
  --config configs/armor_pose/yolov8n/empty10.yaml \
  --output /workspace/runs/armor_pose_custom/train_v1 \
  --epochs 80 --batch 8
```

主要默认值：

| 配置 | 默认值与含义 |
| --- | --- |
| `input_size` | `[480,640]`，固定网络输入 |
| `device` | `'0'`，第一张 CUDA 显卡；可传 `--device cpu` |
| `epochs` / `batch` | 100 / 8 |
| `workers` / `cpu_threads` | 2 / 2；OpenCV 线程数为 1 |
| `amp` | `false`，默认 FP32；可显式 `--amp` |
| 优化器 | AdamW；`lr=0.001`、`weight_decay=0.0005` |
| 学习率 | 3 epoch 线性预热，随后余弦衰减 |
| `patience` | 20，无验证指标提升时提前结束 |
| `brightness` | 0.15；只对训练图片随机乘亮度系数 |
| `val_conf` / `nms_iou` | 0.001 / 0.5，用于验证解码和 AP 统计 |
| `max_train_images` / `max_val_images` | 0 / 0，使用完整 split |

只使用亮度增强；不做色相、饱和度、水平翻转、旋转、mosaic 等增强，避免改变灯光颜色或装甲板数字语义。`flip_idx` 保留点序描述，但当前加载器不会执行翻转。

颜色、类别两条分支使用 sigmoid focal 分类损失；前景和背景项分别归一化。四点回归使用按特征步长缩放后的 Smooth L1，并对四点外接框加 GIoU 损失。匹配使用三层候选的几何先验和 top-k 分配，默认 top-k 为 9。损失权重默认为颜色 1、类别 1、关键点 5、GIoU 1；可以在配置文件的 `loss` 中调整。

默认加载 COCO `yolov8n-pose.pt` 的骨干与 neck，替换并重新初始化检测头，随后联合训练。不会把原教师 ONNX 当成可续训的 Torch checkpoint。若确实要从随机权重开始，可使用 `--from-scratch`。

输出目录非空时新训练会拒绝覆盖。新实验请换一个 `--output`；显存不足时可减小 `--batch`，排查数据加载问题时可设 `--workers 0`。

## 5. 小规模流程检查

先检查数据加载、前向、反向、验证与 checkpoint 保存能否完整运行：

```bash
python src/armor.py train \
  --output /workspace/runs/armor_pose_custom/smoke_check \
  --epochs 1 --batch 8 --workers 0 \
  --max-train-images 64 --max-val-images 32
```

小规模检查会尽量同时取有目标和空标签图片。这里的图片数量限制只用于验证训练流程；模型质量应通过完整训练评估。重新开始完整训练时使用新的输出目录，并保持两个 `max_*_images` 为 0。

## 6. 结果、续训与独立验证

每个实验目录包含：

| 文件 | 用途 |
| --- | --- |
| `args.yaml` | 本次实际使用的配置 |
| `results.csv` | 每个 epoch 的损失、验证指标、学习率和耗时 |
| `metrics.json` | 最近一次验证指标和逐类别结果 |
| `best.pt` | 按验证 mAP50-95 选出的最佳模型，用于推理、验证和导出 |
| `last.pt` | 最近模型，包含优化器、scaler 和随机状态，用于续训 |

从 `last.pt` 续训，`--epochs` 指总训练轮数：

```bash
python src/armor.py train \
  --resume /workspace/runs/armor_pose_custom/train_v1/last.pt \
  --epochs 100
```

续训优先读取 checkpoint 保存的训练配置，默认继续写入该 checkpoint 所在目录；只有显式传入的配置或命令行参数会覆盖。`best.pt` 不保存优化器状态，不能用作 `--resume`。随机状态会恢复，但多 worker 的随机亮度增强不保证逐位复现。

只验证，不进行训练：

```bash
python src/armor.py train \
  --validate /workspace/runs/armor_pose_custom/train_v1/best.pt
```

如果 checkpoint 来自小规模检查，而这次希望评估完整验证集，显式清除数量限制：

```bash
python src/armor.py train \
  --validate /workspace/runs/armor_pose_custom/smoke_check/best.pt \
  --max-val-images 0
```

`mAP50`、`mAP50_95` 根据四点外接框计算，并要求颜色和类别同时匹配。`corner_error_input_px` 是 IoU≥0.5 的正确匹配目标上的平均角点误差，单位为 letterbox 后的网络输入像素；没有匹配时为 `null`。这些不是人体 Pose 的 OKS 指标。验证集没有的类别不会贡献该次 mAP，结合 `per_color_class` 和数据划分报告阅读结果。

## 7. 导出固定接口 ONNX

训练完成后，用已有标注工具虚拟环境中的 ONNX 与 ONNX Runtime 依赖进行导出：

```bash
/workspace/tools/bit-auto-label/.venv/bin/python src/armor.py export \
  --weights /workspace/runs/armor_pose_custom/train_v1/best.pt \
  --output /workspace/runs/armor_pose_custom/train_v1/best.onnx \
  --verify
```

默认在 CPU 上导出。`--verify` 会检查 ONNX 合法性、固定输入输出形状，并用同一输入比较 Torch 和 ONNX Runtime 的输出。ONNX 记录颜色、类别、点序、letterbox 以及网络输入像素坐标等 metadata；导出不包括 NMS。

在图片标注工具中加载新 ONNX 时选择 **letterbox、RGB、除以 255、输入 640×480、输出坐标为网络输入像素**，由工具还原为原图坐标。

## 8. 图片推理

使用本项目训练得到的 `.pt` checkpoint 对图片或图片目录推理：

```bash
python src/armor.py predict \
  --weights /workspace/runs/armor_pose_custom/train_v1/best.pt \
  --source /workspace/data/raw_image \
  --output /workspace/outputs/armor_pose_predictions \
  --conf 0.25 --iou 0.5 --save-images
```

后处理分别选择最高颜色与类别得分，最终置信度为：

```text
confidence = max(color_scores) × max(class_scores)
```

NMS 对相同颜色与类别组合使用四点外接框 IoU。`--conf` 与 `--iou` 是推理后处理参数，不是训练时伪标签生成器的阈值，也不会更改训练标签。输出点会还原为原图像素坐标；查看保存的可视化时同时检查颜色、数字类别和四点点序。

## 9. 统一入口、包目录与学习顺序

从项目根目录运行统一入口，各子命令通过 `--help` 查看参数：

```bash
python src/armor.py --help
python src/armor.py models
python src/armor.py prepare --help
python src/armor.py subsample --help
python src/armor.py train --help
python src/armor.py predict --help
python src/armor.py export --help
```

也可以使用 Python 包入口：

```bash
PYTHONPATH=src python -m armor_pose models
PYTHONPATH=src python -m armor_pose train --config configs/armor_pose/yolov8n/empty10.yaml
```

验证仍通过 `train --validate CHECKPOINT` 执行，没有单独的 `validate` 子命令。

代码和配置目录：

```text
src/
├── armor.py                         # 统一命令入口
└── armor_pose/
    ├── __init__.py
    ├── __main__.py                  # python -m armor_pose
    ├── settings.py                  # 项目根目录及 data/weights/runs 等位置
    ├── common/
    │   ├── definitions.py           # 类别、颜色、点序等共同定义
    │   ├── head.py                  # 4 + 9 + 8 自定义检测头
    │   ├── loss.py                  # 匹配与训练损失
    │   └── runtime.py               # 加载、解码、NMS、验证统计
    ├── data/
    │   ├── dataset.py               # Torch Dataset 与 letterbox
    │   ├── prepare.py               # 导出标签 → 按视频划分的数据集
    │   └── subsample.py             # 分层保留指定比例空图
    ├── models/
    │   ├── __init__.py              # 注册表与模型 factory
    │   ├── yolov8/model.py          # YOLOv8n 适配器
    │   ├── yolov5/model.py          # 原版 YOLOv5n v7.0 适配器
    │   └── yolo26/README.md         # 后续适配说明，尚未实现
    ├── training/
    │   └── engine.py               # 训练、续训与验证
    └── inference/
        ├── predict.py              # 图片推理及原图坐标还原
        └── export.py               # 固定协议 ONNX 导出

configs/
└── armor_pose/
    ├── datasets/armor.yaml
    └── yolov8n/
        ├── default.yaml            # 完整原数据集
        └── empty10.yaml            # 空图保留原数量 10%

tests/
└── armor_pose/                      # 数据、网络、损失、训练及推理测试
```

模块分工如下，路径均相对于 `src/armor_pose/`：

| 模块 | 主要职责 |
| --- | --- |
| `settings.py`、`common/definitions.py` | 目录配置与所有骨干共用的标签、点序定义 |
| `data/prepare.py`、`data/subsample.py` | 数据准备、视频隔离与空图分层抽样 |
| `data/dataset.py` | 图像读取、letterbox、原图点到网络点的转换 |
| `common/head.py`、`common/loss.py` | 共享 21 通道头、匹配和损失 |
| `models/__init__.py`、`models/{yolov8,yolov5}/model.py` | 模型选择与各骨干的特征适配 |
| `training/engine.py` | 训练循环、配置、续训和验证 |
| `common/runtime.py` | 解码、NMS、checkpoint 加载及指标统计 |
| `inference/predict.py`、`inference/export.py` | 图片推理、坐标还原及 ONNX 导出 |

建议按数据流阅读：

1. `common/definitions.py` 和 `settings.py`：确认类别、点序、输入输出约定及目录位置。
2. `data/prepare.py`、`data/subsample.py`、`data/dataset.py`：了解原始 10 列标签、划分、空图抽样和 letterbox 坐标转换。
3. `models/yolov8/model.py` 和 `common/head.py`：从多尺度特征到 21 通道预测。
4. `common/loss.py`：候选匹配、前景与背景监督、四点回归和外接框损失。
5. `training/engine.py`：batch 训练、验证、checkpoint 与续训。
6. `common/runtime.py`、`inference/predict.py`、`inference/export.py`：从模型输出到检测结果和部署模型。

测试从项目根目录运行：

```bash
PYTHONPATH=src python -m unittest discover -s tests -t .
```

旧的五个命令行脚本保留为薄包装，已有启动命令仍可用；实际实现统一位于上面的包目录：

| 旧 CLI | 对应统一命令 |
| --- | --- |
| `src/prepare_pose_dataset.py` | `python src/armor.py prepare` |
| `src/subsample_empty_pose.py` | `python src/armor.py subsample` |
| `src/train_armor_pose.py` | `python src/armor.py train` |
| `src/predict_armor_pose.py` | `python src/armor.py predict` |
| `src/export_armor_pose.py` | `python src/armor.py export` |

旧的顶层核心模块 import 路径不再保留；代码调用请改用 `armor_pose.data`、`armor_pose.models`、`armor_pose.common` 等包路径。旧配置 `configs/train_armor_pose.yaml`、`configs/train_armor_pose_empty10.yaml`、`configs/autoaim_pose.yaml` 分别以相对符号链接指向新的 default、empty10 和数据集配置，保持旧命令与既有配置路径兼容。

源数据、复制后的数据集、权重、ONNX 和运行输出均放在现有忽略目录内。提交代码时只提交源码、配置与说明，保留数据和模型在本地。

## 10. 空标签图片保留原来的 10% 后重训

抽样工具保留所有有目标图片，把空标签图片减少为原数量的 10%，按现有 train/val 和来源视频分层随机选取，固定种子 42。图片和 TXT 逐字节复制到新目录，原始数据集保留。当前 empty10 数据集已经准备完成；需要重新抽样时指定新的 `--output`，或直接使用已有数据训练。

```bash
python src/armor.py subsample --keep-empty 0.1 --seed 42
python src/armor.py train --config configs/armor_pose/yolov8n/empty10.yaml
```

本次从 8989 张空图中保留 899 张，2665 张有目标图片和 3949 个目标全部保留。新数据集共 3564 张：训练 2871 张（正图 2154、空图 717），验证 693 张（正图 511、空图 182）。空图占新数据集约 25.22%；这里的 10% 是相对于原空图数量的保留比例。

数据位于 `datasets/armor_pose_custom_empty10`，抽样明细在其 `subsampling_report.json` 和 `split_manifest.json`。新训练使用本地 COCO 预训练骨干、重新初始化自定义头，配置为 100 轮、batch 8、早停 20；结果写入 `runs/armor_pose_custom/train_empty10`。

验证集也减少了空图，本轮验证 mAP 与旧实验直接对比时评估图片集合不同。正式对比应让两个模型都在原完整验证集上评估，例如对新 checkpoint 执行：

```bash
python src/armor.py train \
  --validate /workspace/runs/armor_pose_custom/train_empty10/best.pt \
  --data /workspace/datasets/armor_pose_custom/data.yaml \
  --max-val-images 0
```

## 11. 接入其他骨干

模型 factory 已实现 `yolov8n-armor-pose`（别名 `yolov8n`）和 `yolov5n-armor-pose`（别名 `yolov5n`）。YOLOv5 使用原版 v7.0 的骨干与 neck。`models/yolo26/` 尚未实现；未知模型名仍明确报错，不会静默改用其他网络。

共享数据、损失、训练和推理的前提是适配器保持相同语义：4 色、9 类、`LT/LB/RB/RT` 四点、RGB letterbox 和网络输入像素坐标。部署输出仍为 `B×21×N`；固定 480×640、步长 8/16/32 时 `N=6300`。训练前向 `return_raw=True` 需要提供：

| raw 字段 | 形状与含义 |
| --- | --- |
| `color_logits` | `(B,N,4)`，未经 sigmoid 的颜色 logits |
| `class_logits` | `(B,N,9)`，未经 sigmoid 的类别 logits |
| `keypoints` | `(B,N,4,2)`，已解码的网络输入像素坐标 |
| `anchor_points` | `(N,2)`，与候选排列一致的训练锚点；损失也接受带 batch 维度 |
| `strides` | `(N,1)`，各候选的步长；损失也接受带 batch 维度 |

接入一个骨干需要完整完成以下步骤：

1. 在对应 `models/<backbone>/` 下实现适配器，加载该骨干本地权重，明确提取的多尺度特征、通道数、步长与候选顺序。
2. 将特征适配到共享检测头，或实现满足同一前向输出与 raw API 的检测头；核实四点解码及训练锚点相互一致。
3. 在 `models/__init__.py` 的 factory 中显式注册模型名，保持模型架构 metadata 和 checkpoint 加载可识别。不能只修改 YAML 的 `model` 字段就宣称完成适配。
4. 在 `configs/armor_pose/<backbone>/` 新增该模型的配置，指定真实可用的预训练权重、`model` 名称和训练参数，继续引用统一数据结构。
5. 在 `tests/armor_pose/` 加入形状、raw API、有限损失、反向梯度、checkpoint 严格加载与 ONNX 一致性测试；先完成小规模训练与推理，再评估完整数据。

如果新骨干改变输入尺寸、特征步长或候选数量，还需要同步适配检测头、部署协议和测试；不能直接沿用当前 `1×21×6300` 的部署假设。


## 13. YOLOv5n 对照实验

使用原版 YOLOv5n v7.0 的 6×6 首层、C3/SPPF 骨干与 PAN neck，输出特征层 17/20/23，对接同一个自定义 21 通道头和损失。比较的是骨干与特征融合网络的差异，数据和训练流程保持一致。

```bash
python src/armor.py train --config configs/armor_pose/yolov5n/empty10.yaml
```

配置使用空图保留原 10% 的数据：训练 2871 张、验证 693 张，100 轮、batch 8、早停 20、seed 42、AdamW lr=0.001。加载本地 `weights/yolov5n.pt` 的 COCO 预训练骨干，严格校验所有骨干张量，重新初始化装甲板头。权重缺失或形状不符会报错；显式 `--from-scratch` 才使用随机骨干。

官方预训练文件来自 [YOLOv5 v7.0](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt)，不是 YOLOv5u。骨干实现和权重来源见 [YOLOv5 说明](armor_pose/models/yolov5/README.md)。

结果保存到 `runs/armor_pose_custom/yolov5n_empty10`。本次后台任务的日志在 `outputs/armor_pose_yolov5n_job/training.log`；任务完成后会自动导出并校验 `best.onnx`，再使用原完整验证集复评，与 YOLOv8n 的同集结果保存在 `comparison_original_val.json`。

手动导出：

```bash
/workspace/tools/bit-auto-label/.venv/bin/python src/armor.py export \
  --weights /workspace/runs/armor_pose_custom/yolov5n_empty10/best.pt --verify
```
