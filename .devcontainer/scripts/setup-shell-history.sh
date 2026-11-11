#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Keep zsh history in the repo so it survives container rebuilds.
history_path="$REPO_ROOT/.devcontainer/.zsh_history"
history_link="$HOME/.zsh_history"

touch "$history_path"
rm -f "$history_link"
ln -s "$history_path" "$history_link"

printf 'Shell history symlinked: %s -> %s\n' "$history_link" "$history_path"
