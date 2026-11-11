# NVIDIA GPU 与 Docker 配置

训练镜像基于 CUDA 13.2，宿主机需要满足：

- NVIDIA 显卡：Turing（RTX 20 / GTX 16 系列）及更新架构。GTX 10 系列及更老的显卡不受镜像内 PyTorch 支持。
- NVIDIA 驱动 **≥ 580**。
- Docker Engine 23.0+，以及 NVIDIA Container Toolkit。

以下步骤均在**主机**（不要在 `docker` 容器）的终端中执行。

### Step 1：安装 NVIDIA 驱动

先检查是否已安装：

```bash
nvidia-smi
```

若右上角 `Driver Version` 已 ≥ 580，跳过本步。否则在 Ubuntu 上执行：

```bash
sudo apt-get update &&
sudo apt-get install -y nvidia-driver-580
```

安装完成后重启，再次运行 `nvidia-smi` 确认。

> 若已安装更高版本（如 590），无需降级。

### Step 2：安装 Docker

参见 [Docker With Proxy](docker_with_proxy.md)。

### Step 3：安装 NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg &&
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list &&
sudo apt-get update &&
sudo apt-get install -y nvidia-container-toolkit &&
sudo nvidia-ctk runtime configure --runtime=docker &&
sudo systemctl restart docker
```

### Step 4：验证

```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

能看到显卡信息即配置完成。

如果报错 `could not select device driver "" with capabilities: [[gpu]]`，说明 Step 3 未生效，重新执行 `nvidia-ctk runtime configure` 并重启 Docker。

## WSL2

- 只在 **Windows** 中安装 NVIDIA 驱动（≥ 580），**不要**在 WSL 里安装 Linux 驱动。
- 在 WSL 中按 Step 2、Step 3 安装 Docker 和 NVIDIA Container Toolkit（或使用开启了 WSL2 后端的 Docker Desktop）。
- 在 WSL 中运行 Step 4 验证。

## 容器内排查

容器内随时可以运行：

```bash
check-env
```

它会打印驱动、PyTorch、Ultralytics 等版本，并在 GPU 上做一次矩阵运算。

| 现象 | 原因 |
| --- | --- |
| `nvidia-smi failed` | 容器没拿到 GPU：Container Toolkit 未安装或 Docker 未重启 |
| `torch.cuda.is_available() is False` | 驱动低于 580 |
| `no kernel image is available` | 显卡架构过老（早于 Turing） |

## References

[Installing the NVIDIA Container Toolkit - NVIDIA Docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
