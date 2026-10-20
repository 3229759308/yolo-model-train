#!/usr/bin/env bash
# 重建独立标注环境；不安装到全局 Python。
set -euo pipefail
ENV_TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "$ENV_TOOL_DIR/../.." && pwd)"
ARMOR_PROJECT="$WORKSPACE_DIR/tools/bit-auto-label"
command -v uv >/dev/null || { echo "需要先安装 uv" >&2; exit 1; }
export UV_CACHE_DIR="$WORKSPACE_DIR/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$WORKSPACE_DIR/.cache/python"
git -C "$WORKSPACE_DIR" -c safe.directory="$WORKSPACE_DIR" -c safe.directory="$ARMOR_PROJECT" submodule update --init --recursive tools/bit-auto-label
uv venv --python 3.13 --seed --allow-existing "$ARMOR_PROJECT/.venv"
uv pip install --python "$ARMOR_PROJECT/.venv/bin/python" torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$ARMOR_PROJECT/.venv/bin/python" -r "$ARMOR_PROJECT/requirements.txt"
# Hugging Face CLI 使用 click；上游业务依赖没有显式包含它。
uv pip install --python "$ARMOR_PROJECT/.venv/bin/python" click==8.5.0
# 上游锁定 NumPy 2.x，按其 README 对 SAM3 使用 --no-deps 安装。
uv pip install --python "$ARMOR_PROJECT/.venv/bin/python" --no-deps -e "$ARMOR_PROJECT/sam3"
echo "完成。运行 bash tools/armorlabel_env/start.sh"
