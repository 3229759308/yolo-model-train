#!/usr/bin/env bash

set -eu

image="yolo-train:latest"

# Without this check compose falls back to pulling docker.io/library/yolo-train and fails with an unclear error.
if ! docker image inspect "$image" >/dev/null 2>&1; then
    printf 'Error: image %s not found on this machine.\n' "$image" >&2
    printf 'Load it from a tarball or registry, or build it from the Dockerfile; see docs/zh-cn/build_docker_image.md.\n' >&2
    exit 1
fi
