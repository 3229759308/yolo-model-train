#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${ARMOR_LABELER_PYTHON:-}" ]]; then
    PYTHON_BIN="$ARMOR_LABELER_PYTHON"
elif [[ -x "$SCRIPT_DIR/../bit-auto-label/.venv/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/../bit-auto-label/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/app.py" "$@"
