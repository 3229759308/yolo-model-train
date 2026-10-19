"""顺序解码视频、按时间采样，保留原始分辨率及帧号清单。"""
import argparse
import csv
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, default=ROOT / 'data/autoaim-1792413441352.webm')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/autoaim_annotation')
    parser.add_argument('--fps', type=float, default=3.0)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error('--fps must be positive')
    cap = cv2.VideoCapture(str(args.video))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or source_fps <= 0:
        raise RuntimeError(f'无法读取视频: {args.video}')
    if args.output.exists():
        raise FileExistsError(f'输出目录已存在，防止覆盖，请使用其他 --output: {args.output}')
    images = args.output / 'images'
    images.mkdir(parents=True)
    frame_index = saved = 0
    next_time = 0.0
    try:
        with (args.output / 'manifest.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'source_video', 'frame_index', 'time_seconds'])
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                seconds = frame_index / source_fps
                if seconds + 1e-9 >= next_time:
                    name = f'{args.video.stem}_f{frame_index:06d}.jpg'
                    if not cv2.imwrite(str(images / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise RuntimeError(f'图片保存失败: {name}')
                    writer.writerow([name, args.video.name, frame_index, f'{seconds:.6f}'])
                    saved += 1
                    next_time = saved / args.fps
                frame_index += 1
    finally:
        cap.release()
    print(f'已读取 {frame_index} 帧，保存 {saved} 张图片: {images}')


if __name__ == '__main__':
    main()
