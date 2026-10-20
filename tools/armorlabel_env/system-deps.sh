#!/usr/bin/env bash
set -euo pipefail
# Docker 容器重建后，以 root 执行一次。
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  libgl1 libegl1 libopengl0 libglib2.0-0 libdbus-1-3 libfontconfig1 \
  libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 \
  libxcb-shape0 libxcb-xinerama0 libxcb-randr0 libxcb-render-util0 \
  libxcb-image0 libxcb-xkb1 xauth fonts-dejavu-core fonts-noto-cjk
