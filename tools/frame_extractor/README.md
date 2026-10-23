# 网页视频抽帧工具

在 `/workspace` 启动（依赖当前容器已安装的 OpenCV，无额外前端依赖）：

```bash
python tools/frame_extractor/app.py
```

浏览器访问 **http://localhost:7860**。如果容器运行在远程机器，在 VS Code 的 Ports 面板转发 7860 端口后打开。
当前 devcontainer 配置使用 host 网络，可直接通过 Docker 宿主机的 localhost:7860 访问。

1. 浏览选择输入文件夹，直接读取该层同一种格式的视频，不递归扫描。
2. 浏览选择输出文件夹，或直接填写新目录路径。
3. 设置帧间隔，例如 30 表示取第 0、30、60… 帧。
4. 点击开始，查看进度；可随时停止，已经生成的图片保留。

所有图片直接写入同一输出文件夹，保留原始分辨率，命名为 `视频名_f0000030.jpg`，不生成 CSV。支持 webm、mp4、mov、avi、mkv、m4v。
已有同视频抽帧图片时拒绝执行，请换新输出目录。停止后重新开始也请选新目录。
进度根据视频标称帧数估计；解码帧数与标称不同时会在日志显示提示。

默认可浏览 `/workspace`；需要其他挂载目录时可指定根目录：

```bash
python tools/frame_extractor/app.py --root /mnt/data --port 7860
```

默认仅监听 127.0.0.1。若使用 Docker bridge 网络且需要映射端口，可加 `--host 0.0.0.0` 并映射 7860。页面选择的是 Docker 可访问的文件夹，宿主机目录需先挂载进容器。

目录中的 `app.py` 为后端，`index.html`、`app.js`、`style.css` 为前端。

## 命令行批量抽帧

无需启动网页，直接运行（依赖 OpenCV）：

```bash
python /workspace/tools/frame_extractor/batch_extract.py
```

默认递归扫描 `/workspace/data/infantry` 和 `/workspace/data/sentry` 的 AVI，
抽取第 0、30、60… 帧，以原始分辨率、95 的 JPG 质量保存到 `/workspace/data/raw_image`。
文件名如 `infantry_bisai_1_f0000030.jpg`、`sentry_bisai_1_f0000030.jpg`，
包括来源目录、视频名与从 0 开始的原视频帧序号。子目录名也会加入文件名；重名冲突会在处理前报错。
默认最多同时处理 4 个视频，已有非空图片会跳过，不覆盖；单个视频失败会继续其他视频，最终返回非零退出码。
部分 AVI 的标称帧数为 0，仍会顺序读取至视频结束。

指定输入、输出及间隔：

```bash
python /workspace/tools/frame_extractor/batch_extract.py \
  --input /workspace/data/infantry /workspace/data/sentry \
  --output /workspace/data/raw_image --step 30 --workers 4 --quality 95
```
