#!/usr/bin/env python3
"""Docker 内的本地视频抽帧网页工具。"""
import argparse
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXTENSIONS = {'.webm', '.mp4', '.mov', '.avi', '.mkv', '.m4v'}
LOCK = threading.RLock()
CANCEL = threading.Event()
STATE = dict(status='idle', saved=0, completed=0, total=0, progress=0, current='', logs=[], output='')


def update(**values):
    with LOCK:
        STATE.update(values)


def log(message):
    with LOCK:
        STATE['logs'] = (STATE['logs'] + [message])[-100:]


def folder(value):
    path = Path(value).expanduser()
    path = (ROOT / path if not path.is_absolute() else path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'请选择 {ROOT} 内的文件夹')
    return path


def videos_in(path):
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)


def run(videos, output, step):
    saved = 0
    try:
        output.mkdir(parents=True, exist_ok=True)
        for index, video in enumerate(videos):
            if CANCEL.is_set():
                break
            update(current=video.name)
            log(f'正在处理 {video.name}')
            cap = cv2.VideoCapture(str(video))
            count = local_saved = 0
            try:
                if not cap.isOpened():
                    raise ValueError(f'无法打开视频：{video.name}')
                estimate = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                expected = int(estimate) if math.isfinite(estimate) and estimate > 0 else 0
                while not CANCEL.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if count % step == 0:
                        name = f'{video.stem}_f{count:07d}.jpg'
                        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        if not ok:
                            raise ValueError(f'图片编码失败：{name}')
                        with (output / name).open('xb') as f:
                            f.write(encoded.tobytes())
                        saved += 1
                        local_saved += 1
                    count += 1
                    if count % 20 == 0 or count == 1:
                        progress = (index + min(count / expected, .99) if expected else index) / len(videos) * 100
                        update(saved=saved, progress=round(progress, 1))
                if CANCEL.is_set():
                    log(f'已停止，保留已生成的 {saved} 张图片。')
                    break
                if count == 0:
                    raise ValueError(f'视频没有可解码帧：{video.name}')
                log(f'{video.name}：读取 {count} 帧，保存 {local_saved} 张')
                if expected and expected != count:
                    log(f'提示：标称 {expected} 帧，实际解码 {count} 帧；请检查原视频完整性。')
                update(completed=index + 1, progress=round((index + 1) / len(videos) * 100, 1))
            finally:
                cap.release()
        update(status='cancelled' if CANCEL.is_set() else 'done', saved=saved)
        if not CANCEL.is_set():
            log(f'全部完成，共保存 {saved} 张图片。')
    except Exception as exc:
        log(f'处理失败：{exc}；已生成的图片保留。')
        update(status='error', saved=saved)


def start(data):
    with LOCK:
        if STATE['status'] == 'running':
            raise ValueError('已有任务正在运行')
        source, output = folder(data['input']), folder(data['output'])
        step = data.get('step')
        if type(step) is not int or step < 1:
            raise ValueError('帧间隔必须是大于 0 的整数')
        if not source.is_dir():
            raise ValueError('输入文件夹不存在')
        if output.exists() and not output.is_dir():
            raise ValueError('输出路径不是文件夹')
        videos = videos_in(source)
        if not videos:
            raise ValueError('输入文件夹中没有支持的视频（不扫描子文件夹）')
        if len({p.suffix.lower() for p in videos}) != 1:
            raise ValueError('输入文件夹包含多种视频格式，请放入同一种格式的视频')
        stems = {p.stem for p in videos}
        if len(stems) != len(videos):
            raise ValueError('视频名称重复，请先重命名')
        if output.is_dir() and any(p.is_file() and p.suffix.lower() == '.jpg' and any(p.name.startswith(s + '_f') for s in stems) for p in output.iterdir()):
            raise ValueError('输出目录已有这些视频的抽帧图片，请选择新目录，避免覆盖')
        CANCEL.clear()
        update(status='running', saved=0, completed=0, total=len(videos), progress=0,
               current='', output=str(output), logs=[f'准备处理 {len(videos)} 个视频，每隔 {step} 帧取一张。'])
        threading.Thread(target=run, args=(videos, output, step), daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    def reply(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == '/api/status':
                with LOCK:
                    self.reply(STATE)
            elif url.path == '/api/folders':
                path = folder(parse_qs(url.query).get('path', [str(ROOT)])[0])
                if not path.is_dir():
                    raise ValueError('文件夹不存在')
                children = sorted(p.name for p in path.iterdir() if p.is_dir() and not p.name.startswith('.') and p.resolve().is_relative_to(ROOT))
                videos = videos_in(path)
                self.reply(dict(path=str(path), root=str(ROOT), parent=str(path.parent) if path != ROOT else None,
                                folders=children, videos=[dict(name=p.name, bytes=p.stat().st_size) for p in videos]))
            elif url.path == '/api/config':
                default = ROOT / 'data/autoaim_old'
                self.reply(dict(root=str(ROOT), input=str(default / 'row_video'), output=str(default / 'extractive_image')))
            elif url.path in ('/', '/app.js', '/style.css'):
                name = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}[url.path]
                body = (HERE / name).read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', {'/': 'text/html; charset=utf-8', '/app.js': 'text/javascript; charset=utf-8', '/style.css': 'text/css; charset=utf-8'}[url.path])
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.reply({'error': 'Not found'}, 404)
        except (ValueError, OSError) as exc:
            self.reply({'error': str(exc)}, 400)

    def do_POST(self):
        try:
            origin = self.headers.get('Origin')
            if origin and urlparse(origin).netloc != self.headers.get('Host'):
                self.reply({'error': '不允许跨站请求'}, 403)
                return
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                raise ValueError('需要 JSON 请求')
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 16384:
                raise ValueError('请求大小无效')
            data = json.loads(self.rfile.read(size))
            if self.path == '/api/start':
                start(data)
            elif self.path == '/api/stop':
                with LOCK:
                    if STATE['status'] == 'running':
                        CANCEL.set()
            else:
                self.reply({'error': 'Not found'}, 404)
                return
            self.reply({'ok': True})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.reply({'error': str(exc)}, 400)

    def log_message(self, fmt, *args):
        pass


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--root', type=Path, default=ROOT, help='可浏览的文件夹根目录，默认工作区')
    args = parser.parse_args()
    ROOT = args.root.resolve()
    if not ROOT.is_dir():
        parser.error('--root 必须是存在的文件夹')
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'抽帧工具已启动：http://{args.host}:{server.server_port}  文件范围：{ROOT}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        CANCEL.set()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
