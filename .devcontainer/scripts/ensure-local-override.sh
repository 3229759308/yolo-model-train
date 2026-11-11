#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
override_file="$SCRIPT_DIR/../docker-compose.local.yml"

# docker-compose.local.yml holds per-machine mounts and is not committed.
# devcontainer.json always lists it, so create an empty one when it is missing.
if [ ! -f "$override_file" ]; then
    cat > "$override_file" <<'EOF'
# Per-machine overrides for this dev container; not committed.
# Example: share the host opencode config and sessions with the container.
#
# services:
#   yolo-train:
#     volumes:
#       - type: bind
#         source: ${HOME}/.config/opencode
#         target: /home/ubuntu/.config/opencode
#       - type: bind
#         source: ${HOME}/.local/share/opencode
#         target: /home/ubuntu/.local/share/opencode
services:
  yolo-train: {}
EOF
    printf 'Created empty %s\n' "$override_file"
fi
