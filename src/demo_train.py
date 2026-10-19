"""下载 COCO8，训练 YOLO11n 一轮，然后保存验证集预测图片。"""
from pathlib import Path

import torch
import yaml
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.downloads import safe_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    dataset_dir = ROOT / "datasets"
    if not (dataset_dir / "coco8/images/train").is_dir():
        safe_download(
            url="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip",
            dir=dataset_dir,
            unzip=True,
            delete=True,
        )
    original = Path(ultralytics.__file__).parent / "cfg/datasets/coco8.yaml"
    data = yaml.safe_load(original.read_text())
    data["path"] = str(dataset_dir / "coco8")
    data.pop("download", None)
    data_file = dataset_dir / "coco8-demo.yaml"
    data_file.write_text(yaml.safe_dump(data, allow_unicode=True))
    config = yaml.safe_load((ROOT / "configs/demo.yaml").read_text())
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"训练设备: {device}", flush=True)
    model = YOLO(str(ROOT / "weights/yolo11n.pt"))
    model.train(
        data=str(data_file), device=device,
        project=str(ROOT / "runs"), name="demo_train", **config,
    )
    best = Path(model.trainer.save_dir) / "weights/best.pt"
    trained = YOLO(str(best))
    trained.predict(
        source=str(dataset_dir / "coco8/images/val"),
        device=device, imgsz=config["imgsz"], save=True,
        project=str(ROOT / "runs"), name="demo_predict",
    )
    print(f"训练权重: {best}")
    print(f"预测图片: {trained.predictor.save_dir}")


if __name__ == "__main__":
    main()
