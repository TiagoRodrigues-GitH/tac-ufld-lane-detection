#!/usr/bin/env bash
# Measure real latency / memory ON THE JETSON (replaces the simulated slowdown).
#
#   ./scripts/jetson/measure_on_jetson.sh <checkpoint.pt> <deployment_dir> <video.mp4> [fp16]
#
# Prerequisites: JetPack with TensorRT; PyTorch for JetPack (NVIDIA wheel);
# `pip install -e ".[deploy]"` (onnx/onnxruntime are only needed for export).
set -euo pipefail
CKPT="${1:?checkpoint}"; DEPLOY="${2:?deployment dir}"; VIDEO="${3:?video}"; PRECISION="${4:-fp16}"

sudo nvpmodel -q || true                     # record the power mode used
sudo jetson_clocks --show || true            # record the clocks (run `sudo jetson_clocks` for max clocks)
OUT="results/benchmarks/jetson_$(hostname)_${PRECISION}"
tegrastats --interval 500 --logfile "$OUT.tegrastats.log" &
TS=$!
trap 'kill $TS 2>/dev/null || true' EXIT

python3 -m tac_ufld benchmark --checkpoints "$CKPT" --backend tensorrt --deployments "$DEPLOY" \
  --precision "$PRECISION" --video "$VIDEO" --profile configs/deploy/desktop.yaml --slowdown 1.0 \
  --measure-frames 300 --out "$OUT"
echo "report: $OUT.md ; memory/power trace: $OUT.tegrastats.log"
