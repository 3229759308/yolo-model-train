#!/usr/bin/env bash
set -euo pipefail
ENV_TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "$ENV_TOOL_DIR/../.." && pwd)"
ARMOR_PYTHON="$WORKSPACE_DIR/tools/bit-auto-label/.venv/bin/python"
if [[ ! -x "$ARMOR_PYTHON" ]]; then
  echo "标注环境不存在，请先执行 bash tools/armorlabel_env/setup.sh" >&2
  exit 1
fi
exec "$ARMOR_PYTHON" "$ENV_TOOL_DIR/launch.py" "$@"
