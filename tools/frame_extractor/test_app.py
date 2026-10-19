"""运行：python -m unittest discover -s tools/frame_extractor -p 'test_*.py'。"""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

import app


class AppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='frame_web_')
        cls.root = Path(cls.temp.name)
        app.ROOT = cls.root
        source = cls.root / 'videos'
        source.mkdir()
        for name in ('first', 'second'):
            writer = cv2.VideoWriter(str(source / (name + '.avi')), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
            if not writer.isOpened():
                raise RuntimeError('测试视频编码器不可用')
            for index in range(21):
                writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
            writer.release()
        cls.server = app.ThreadingHTTPServer(('127.0.0.1', 0), app.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.temp.cleanup()

    def request(self, path, data=None):
        req = Request(self.url + path, data=json.dumps(data).encode() if data is not None else None,
                      headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=5) as response:
            return json.load(response)

    def test_workflow(self):
        for path in ('/', '/style.css', '/app.js'):
            with urlopen(self.url + path, timeout=5) as response:
                self.assertEqual(response.status, 200)
                self.assertGreater(len(response.read()), 100)
        folders = self.request('/api/folders?path=videos')
        self.assertEqual(len(folders['videos']), 2)
        output = self.root / 'output'
        data = {'input': str(self.root / 'videos'), 'output': str(output), 'step': 7}
        self.request('/api/start', data)
        for _ in range(200):
            state = self.request('/api/status')
            if state['status'] != 'running':
                break
            time.sleep(.02)
        self.assertEqual(state['status'], 'done', state)
        self.assertEqual(state['saved'], 6)
        self.assertEqual(state['completed'], 2)
        self.assertEqual(state['progress'], 100)
        self.assertEqual(sorted(p.name for p in output.iterdir()), [f'{name}_f{i:07d}.jpg' for name in ('first', 'second') for i in (0, 7, 14)])
        self.assertTrue(all(cv2.imread(str(p)).shape == (48, 64, 3) for p in output.iterdir()))
        with self.assertRaises(HTTPError) as conflict:
            self.request('/api/start', data)
        self.assertEqual(conflict.exception.code, 400)
        with self.assertRaises(HTTPError):
            self.request('/api/start', {**data, 'step': 0})
        with self.assertRaises(HTTPError):
            self.request('/api/folders?path=/')
        self.assertEqual(len(list(output.iterdir())), 6)
        self.assertTrue(self.request('/api/stop', {})['ok'])


if __name__ == '__main__':
    unittest.main()
