#!/bin/sh
# Fills the Docker volume openlane-local once (2026-10-22): the OpenLane labels copied to the Linux side and every
# image decoded at the network size (scripts/cache_images.py, 480x320 JPEG q95), so the training container reads
# neither 1920x1280 JPEGs nor ~200k JSON files through the Windows bind mount. Safe to stop and rerun: the label
# copy is skipped when complete and the cache script skips files that exist.
#
#   docker compose -f docker-compose.openlane.yml run --rm --no-deps \
#     -v ./configs:/app/configs:ro -v ./scripts:/app/scripts:ro \
#     openlane-full sh scripts/docker/fill_openlane_volume.sh
set -eu
SRC=/data/openlane
DST=/data/openlane_local
WORKERS="${WORKERS:-6}"

if [ ! -f "$DST/lane3d_1000/.complete" ]; then
  echo "$(date '+%F %T') copying labels $SRC/lane3d_1000 -> $DST/lane3d_1000"
  rm -rf "$DST/lane3d_1000.part"
  cp -a "$SRC/lane3d_1000" "$DST/lane3d_1000.part"
  rm -rf "$DST/lane3d_1000"
  mv "$DST/lane3d_1000.part" "$DST/lane3d_1000"
  touch "$DST/lane3d_1000/.complete"
fi
echo "$(date '+%F %T') labels: $(find "$DST/lane3d_1000" -name '*.json' | wc -l) json files, $(du -sh "$DST/lane3d_1000" | cut -f1)"

echo "$(date '+%F %T') image cache -> $DST/cache ($WORKERS workers)"
python scripts/cache_images.py --config configs/openlane_full_7models.yaml --workers "$WORKERS"
echo "$(date '+%F %T') done: $(du -sh "$DST/cache" | cut -f1)"
