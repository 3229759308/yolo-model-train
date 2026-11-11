#!/bin/bash

: "${YOLO_PATH:=/workspace}"
export YOLO_PATH

export PATH="${PATH}:${YOLO_PATH}/.script"
