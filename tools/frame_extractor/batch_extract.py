#!/usr/bin/env python3
"""将多个目录中的 AVI 按帧间隔批量抽取为 JPG，默认处理 infantry 和 sentry。"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import cv2


ROOT = Path(__file__).resolve().parents[2]


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def find_videos(sources):
    """递归收集 AVI，并在写入前检查扁平输出文件名是否冲突。"""
    jobs = []
    seen_videos = set()
    prefixes = {}
    for source in sources:
        source = source.expanduser().resolve()
        if not source.is_dir():
            raise ValueError(f"输入文件夹不存在：{source}")
        for video in sorted(source.rglob("*")):
            if not video.is_file() or video.suffix.lower() != ".avi":
                continue
            resolved = video.resolve()
            if resolved in seen_videos:
                continue
            seen_videos.add(resolved)
            relative = video.relative_to(source).with_suffix("")
            prefix = "_".join((source.name, *relative.parts))
            if prefix in prefixes:
                raise ValueError(f"输出名称冲突，请重命名：{prefixes[prefix]} 和 {video}")
            prefixes[prefix] = video
            jobs.append((video, prefix))
    if not jobs:
        raise ValueError("输入文件夹中没有找到 AVI 视频")
    return jobs


def extract_video(video, prefix, output, step, quality):
    cap = cv2.VideoCapture(str(video))
    count = saved = skipped = 0
    last_progress = time.monotonic()
    try:
        if not cap.isOpened():
            raise RuntimeError("无法打开视频")
        estimate = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        expected = int(estimate) if math.isfinite(estimate) and estimate > 0 else 0
        print(f"[开始] {video}", flush=True)
        # 部分 AVI 的标称帧数为 0；顺序读取至结束，不按标称帧数停止。
        # grab 跳过非采样帧的颜色转换，保留准确的帧序号。
        while cap.grab():
            if count % step == 0:
                target = output / f"{prefix}_f{count:07d}.jpg"
                if target.exists():
                    if not target.is_file() or target.stat().st_size == 0:
                        raise RuntimeError(f"已有输出不是有效文件，请检查：{target}")
                    skipped += 1
                else:
                    ok, frame = cap.retrieve()
                    if not ok or frame is None:
                        raise RuntimeError(f"无法解码第 {count} 帧")
                    ok, encoded = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
                    )
                    if not ok:
                        raise RuntimeError(f"第 {count} 帧 JPG 编码失败")
                    # 先完整写入临时文件，再以硬链接原子创建最终文件。
                    # 中断不会留下被下一次运行误跳过的残缺 JPG；已有文件不会被覆盖。
                    with tempfile.NamedTemporaryFile(dir=output, suffix=".tmp") as stream:
                        stream.write(encoded.tobytes())
                        stream.flush()
                        os.fchmod(stream.fileno(), 0o644)
                        os.link(stream.name, target)
                    saved += 1
            count += 1
            if time.monotonic() - last_progress >= 15:
                print(f"[进度] {prefix}：已读取 {count} 帧，新增 {saved} 张，跳过 {skipped} 张", flush=True)
                last_progress = time.monotonic()
        if count == 0:
            raise RuntimeError("视频没有可读取的帧")
        if expected and expected != count:
            print(f"[提示] {video}：标称 {expected} 帧，实际读取 {count} 帧，请检查原视频完整性。", flush=True)
    except Exception as exc:
        return count, saved, skipped, str(exc)
    finally:
        cap.release()
    return count, saved, skipped, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", nargs="+", type=Path,
        default=[ROOT / "data/infantry", ROOT / "data/sentry"],
        help="一个或多个输入目录，递归扫描 AVI（含大写扩展名）",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw_image")
    parser.add_argument("--step", type=positive_int, default=30, help="帧间隔，默认 30，从第 0 帧开始")
    parser.add_argument("--quality", type=int, default=95, help="JPG 质量 1–100，默认 95")
    parser.add_argument("--workers", type=positive_int, default=min(4, os.cpu_count() or 1), help="并行处理视频数量，默认最多 4 个")
    args = parser.parse_args(argv)
    if not 1 <= args.quality <= 100:
        parser.error("--quality 必须在 1–100 之间")
    try:
        jobs = find_videos(args.input)
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    cv2.setNumThreads(1)
    print(f"找到 {len(jobs)} 个 AVI，每隔 {args.step} 帧保存一张 JPG；输出：{output}", flush=True)
    total_saved = total_skipped = failed = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = {
            pool.submit(extract_video, video, prefix, output, args.step, args.quality): video
            for video, prefix in jobs
        }
        for completed, future in enumerate(as_completed(futures), 1):
            video = futures[future]
            try:
                count, saved, skipped, error = future.result()
            except Exception as exc:
                count = saved = skipped = 0
                error = str(exc)
            total_saved += saved
            total_skipped += skipped
            print(f"[{completed}/{len(jobs)}] {video.parent.name}/{video.name}：读取 {count} 帧，新增 {saved} 张，跳过 {skipped} 张", flush=True)
            if error:
                failed += 1
                print(f"[失败] {video}：{error}", file=sys.stderr, flush=True)
    print(f"处理结束：新增 {total_saved} 张，跳过 {total_skipped} 张，失败 {failed} 个视频；耗时 {time.monotonic() - started:.1f} 秒。", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
