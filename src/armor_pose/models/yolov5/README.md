# 原版 YOLOv5n v7.0 自定义装甲板适配器

模型名为 `yolov5n-armor-pose`，别名 `yolov5n`。实现对照 [官方 v7.0 yolov5n.yaml](https://github.com/ultralytics/yolov5/blob/v7.0/models/yolov5n.yaml)：6×6/stride2/pad2 首层，depth=0.33、width=0.25，C3 重复数 1/2/3/1，SPPF 与 PAN neck，P3/P4/P5 层为 17/20/23、通道为 64/128/256。

这份实现保持原版骨干计算，SPPF 的 cv1 显式使用 SiLU，骨干 BN eps=0.001、momentum=0.03。共享装甲板头保持原有设置，输出独立 4 色、9 类和 LT/LB/RB/RT 四点，固定 480×640 输入下输出 `1×21×6300`。

预训练权重只从本地可信的 [官方 v7.0 yolov5n.pt](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt) 加载，COCO Detect 层丢弃。原始文件 4,062,133 字节，SHA256：

```text
4f180cf23ba0717ada0badd6c685026d73d48f184d00fc159c2641284b2ac0a3
```

兼容读取原版 checkpoint 的类名，无需引入原训练脚本的日志/下载功能；对全部 342 个骨干 state 张量严格校验并加载。训练生成的本项目 checkpoint 使用纯 state_dict 与 architecture metadata，可在 `pretrained=False` 下离线重建。

训练入口：

```bash
python src/armor.py train --config configs/armor_pose/yolov5n/empty10.yaml
```

骨干原始预训练加载已与官方模型的三层特征逐元素比对。检测头、标签定义、四点损失与原图坐标恢复共用 `armor_pose.common` 和 `armor_pose.data`。
