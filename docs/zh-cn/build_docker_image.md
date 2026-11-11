# Build Docker Image

开发容器直接使用本机已有的 `yolo-train:latest` 镜像，打开容器时**不会**自动构建。镜像可以通过以下任一方式获得：

- 自行构建（维护者，或修改了 `Dockerfile` 时）
- 从镜像仓库拉取
- 从离线文件导入

## 构建

于 Repo 根目录下，在终端中使用：

```bash
docker build . --target yolo-train -t yolo-train:latest
```

首次构建需要下载约 10 GB 的 `ultralytics/ultralytics` 基础镜像，之后增量构建只需几分钟。

构建完成后，在 VSCode 中执行 `Dev Containers: Rebuild Container` 即可切换到新镜像。

### 代理构建

部分国家和地区会阻断对 `github` 和 `pypi` 的连接，此时构建需要使用代理：

```bash
docker build . --target yolo-train -t yolo-train:latest \
  --network host \
  --build-arg HTTP_PROXY=http://127.0.0.1:7890 \
  --build-arg HTTPS_PROXY=http://127.0.0.1:7890
```

请自行把 `http://127.0.0.1:7890` 改为合适的代理地址。

如果不使用本机回环地址 (`127.0.0.1`) 作为代理地址，构建参数中的 `--network host` 可以省去。

### 基础镜像版本

`Dockerfile` 通过 digest 锁定了基础镜像（Ubuntu 24.04、Python 3.12、PyTorch 2.14.1 + CUDA 13.2、Ultralytics 8.4.182），保证每台机器得到相同的环境。

如需升级，查询新的 digest：

```bash
docker buildx imagetools inspect ultralytics/ultralytics:latest | grep -m1 Digest
```

然后修改 `Dockerfile` 中的 `BASE_IMAGE`，或临时用构建参数覆盖：

```bash
docker build . --target yolo-train -t yolo-train:latest \
  --build-arg BASE_IMAGE=ultralytics/ultralytics:latest
```

升级后在容器中运行 `check-env` 和单元测试确认环境可用。

## 分发

镜像约 12 GB，接收方只需要镜像和本仓库代码，**不需要**重新构建，也不需要访问 `github` / `pypi`。

接收方拿到镜像后，按 [README](../../README.md#开发环境) 克隆仓库并 `Reopen in Container` 即可。

### 方式一：镜像仓库（推荐长期使用）

可以使用 Docker Hub，或国内访问更快的 [阿里云容器镜像服务](https://cr.console.aliyun.com/)（个人版免费）。以下用 `<仓库地址>` 代指仓库中的镜像名，例如：

- Docker Hub：`<用户名>/yolo-train`
- 阿里云：`registry.cn-hangzhou.aliyuncs.com/<命名空间>/yolo-train`

维护者构建后推送：

```bash
docker login <仓库域名>
docker tag yolo-train:latest <仓库地址>:latest
docker push <仓库地址>:latest
```

接收方拉取，并改名为开发容器使用的 `yolo-train:latest`：

```bash
docker pull <仓库地址>:latest
docker tag <仓库地址>:latest yolo-train:latest
```

> 私有仓库需要接收方先 `docker login`。

### 方式二：离线文件

在已构建的机器上导出：

```bash
docker save yolo-train:latest | gzip > yolo-train.tar.gz
```

使用任意方式（如 U 盘、scp、网盘）拷贝到目标机器后导入：

```bash
docker load -i yolo-train.tar.gz
```

导入后镜像名即为 `yolo-train:latest`，无需改名。

### 更新镜像

修改 `Dockerfile` 后，维护者重新构建并按上述方式重新分发；接收方拉取或导入新镜像后，执行 `Dev Containers: Rebuild Container`。
