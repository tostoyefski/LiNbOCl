#!/bin/bash
set -euo pipefail
repo=/home/litao/projects/LiNbOCl
runtime=/home/litao/.local/share/crystal-flow-udocker
rootfs="$runtime/containers/crystal-flow-py312/ROOT"
export PROOT_NO_SECCOMP=1 PYTHONDONTWRITEBYTECODE=1
exec "$runtime/bin/proot-x86_64" --kill-on-exit -r "$rootfs" \
 -b /dev -b /proc -b /sys -b /home/litao/projects -b /etc/resolv.conf \
 -b "$repo/_runtime/tmp:/tmp" -b "$repo/_runtime/home:/root" \
 -w "$repo" /usr/bin/env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "$@"
