#!/usr/bin/env bash
# OpenLane-V input data: labels and lists (RVLD, ICCV 2023) + the cropped 640x384 frames, streamed from the
# OpenLane image archives without extracting them (~6 GB written; the archives, ~103 GB, are read over HTTP).
#   OPENLANE_ROOT=/scratch/$USER/OpenLane bash scripts/cluster/prepare_openlane_v.sh
# Rerunnable: finished archives are skipped. Takes ~1-3 h depending on the network (CPU node is enough).
set -euo pipefail
: "${OPENLANE_ROOT:?set OPENLANE_ROOT to a folder with ~10 GB free}"
. .venv/bin/activate
mkdir -p "$OPENLANE_ROOT"
if [ ! -d "$OPENLANE_ROOT/OpenLane-V/label" ]; then
  # OpenLane-V labels and lists, Google Drive file of the RVLD repository (github.com/dongkwonjin/RVLD)
  python -m gdown 1Jf7g1EG2oL9uVi9a1Fk80Iqtd1Bvb0V7 -O "$OPENLANE_ROOT/OpenLane-V.zip"
  unzip -q "$OPENLANE_ROOT/OpenLane-V.zip" -d "$OPENLANE_ROOT" && rm "$OPENLANE_ROOT/OpenLane-V.zip"
fi
python scripts/stream_openlane_v_cache.py "$OPENLANE_ROOT" --workers 8
