#!/usr/bin/env bash
# Build TensorRT engines ON THE JETSON from an exported deployment folder.
# Engines are specific to the GPU, the TensorRT version and the JetPack
# release: never copy engines built on a desktop to a Jetson.
#
#   ./scripts/jetson/build_engines.sh <deployment_dir> [fp16|fp32|int8] [workspace_MB]
#
# <deployment_dir> comes from `python -m tac_ufld export` (desktop) and holds
# deployment.json, the FP32 ONNX graphs, fp16/ (FP16 ONNX) and int8/ (Q/DQ ONNX).
# TensorRT >= 10 builds strongly typed networks from these files; with an older
# JetPack (TensorRT 8.x) use the FP32 ONNX and the --fp16 / --int8 flags instead.
set -euo pipefail

DEPLOY="${1:?usage: build_engines.sh <deployment_dir> [fp16|fp32|int8] [workspace_MB]}"
PRECISION="${2:-fp16}"
WORKSPACE="${3:-1024}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

case "$PRECISION" in
  fp32) SRC="$DEPLOY" ;;
  fp16) SRC="$DEPLOY/fp16" ;;
  int8) SRC="$DEPLOY/int8" ;;
  *) echo "precision must be fp32, fp16 or int8" >&2; exit 2 ;;
esac
[ -f "$SRC/deployment.json" ] || { echo "missing $SRC/deployment.json (export with --fp16 / --int8 first)" >&2; exit 2; }
[ -x "$TRTEXEC" ] || { echo "trtexec not found at $TRTEXEC (install JetPack's TensorRT)" >&2; exit 2; }

mkdir -p "$DEPLOY/engines"
TRT_VERSION=$(dpkg-query -W -f='${Version}' tensorrt 2>/dev/null || echo unknown)
L4T=$(head -n 1 /etc/nv_tegra_release 2>/dev/null || echo unknown)

for onnx in "$SRC"/*.onnx; do
  name=$(basename "$onnx" .onnx)
  [ "$name" = "clip" ] && [ "$PRECISION" != "fp32" ] && continue   # the clip graph is a non-streaming reference
  engine="$DEPLOY/engines/$name.$PRECISION.engine"
  echo "== $name ($PRECISION) -> $engine"
  extra=""
  [ "$PRECISION" = "fp32" ] && extra="--noTF32"
  "$TRTEXEC" --onnx="$onnx" --saveEngine="$engine" --memPoolSize=workspace:"$WORKSPACE" $extra \
    --skipInference --verbose=false 2>&1 | tail -n 5
done

cat > "$DEPLOY/engines/build_info.$PRECISION.jetson.txt" <<EOF
built_on: $(hostname)
l4t: $L4T
tensorrt: $TRT_VERSION
precision: $PRECISION
date: $(date -Iseconds)
EOF
echo "done; engines in $DEPLOY/engines"
