# Docker Engine 23.0+ (BuildKit) is required.

# ultralytics/ultralytics:latest as of 2026-10-19: Ubuntu 24.04, Python 3.12,
# torch 2.14.1+cu132, torchvision 0.29.1, ultralytics 8.4.182, opencv 5.0 (headless).
# CUDA 13.2 needs NVIDIA driver >= 580 and a Turing (sm_75, RTX 20 series) or newer GPU.
# The `latest` tag moves daily; keep the digest so every machine gets the same stack.
ARG BASE_IMAGE=ultralytics/ultralytics@sha256:7073b36dc3d48d3b906c22c70ad57a552ec1a4c4cbc48eb786ecea12d010f345

FROM ${BASE_IMAGE} AS yolo-train

SHELL ["/bin/bash", "-c"]

ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive

# Shell and everyday tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    zsh tmux vim less sudo \
    git git-lfs openssh-client \
    tree ripgrep && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

# Qt/X11 runtime and CJK fonts for the PySide6 labeling tool
COPY tools/armor_labeler/system-deps.sh /tmp/system-deps.sh
RUN bash /tmp/system-deps.sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

# Extra Python packages used by src/ and tools/ on top of the base image.
# onnxruntime-gpu is the CUDA 13 build and loads the CUDA/cuDNN libraries installed with torch.
# It provides the same module as the CPU onnxruntime package, so the two must never coexist.
RUN pip uninstall -y onnxruntime && \
    pip install --no-cache-dir \
    onnx==1.23.2 \
    onnxslim==0.1.99 \
    onnxruntime-gpu==1.31.1 \
    PySide6-Essentials==6.12.0 \
    pytest==9.2.0

# Change user
RUN chsh -s /bin/zsh ubuntu && \
    echo "ubuntu ALL=(ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers

# Precreate XDG-style parent directories for bind mounts and tool caches under ubuntu's home.
RUN mkdir -p \
        /home/ubuntu/.cache \
        /home/ubuntu/.config \
        /home/ubuntu/.local/share \
        /home/ubuntu/.local/state && \
    chown -R ubuntu:ubuntu /home/ubuntu
WORKDIR /home/ubuntu
ENV USER=ubuntu
USER ubuntu

# Install oh my zsh, change theme to af-magic and setup environment of zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    sed -i 's/^ZSH_THEME=.*/ZSH_THEME="af-magic"/' ~/.zshrc && \
    sed -i "s/^# zstyle ':omz:update' mode disabled/zstyle ':omz:update' mode disabled/" ~/.zshrc && \
    echo '# Hint: uncomment and set YOLO_PATH if the repo is not mounted at /workspace.' >> ~/.zshrc && \
    echo '# export YOLO_PATH="/workspace"' >> ~/.zshrc && \
    echo 'source ~/env_setup.zsh' >> ~/.zshrc && \
    echo 'source ~/env_setup.bash' >> ~/.bashrc

# Install opencode, pinned to the version verified with the team's provider config
ARG OPENCODE_VERSION=2.2.5
RUN curl -fsSL https://opencode.ai/v2/install | bash -s -- --version "${OPENCODE_VERSION}" --no-modify-path && \
    echo '# The container shares the host network, where the host opencode service already holds its port;' >> ~/.zshrc && \
    echo '# run a private server inside the container instead.' >> ~/.zshrc && \
    echo 'alias opencode="opencode --standalone"' >> ~/.zshrc && \
    echo 'alias opencode="opencode --standalone"' >> ~/.bashrc
ENV PATH="/home/ubuntu/.opencode/bin:${PATH}"

# Copy environment setup scripts
COPY --chown=1000:1000 .script/template/env_setup.bash env_setup.bash
COPY --chown=1000:1000 .script/template/env_setup.zsh env_setup.zsh

WORKDIR /workspace
