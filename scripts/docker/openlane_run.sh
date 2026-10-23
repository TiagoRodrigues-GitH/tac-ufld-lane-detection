#!/bin/sh
# Entry point of the docker-compose.openlane.yml services: one OpenLane run, safe to restart at any time.
# RUN_NAME selects the run and its config (configs/$RUN_NAME.yaml); default openlane_7models.
#
# * Restart after a shutdown or a crash: Docker restarts the container (restart: unless-stopped) and `--resume`
#   continues from the last finished epoch (at most one epoch is repeated).
# * GPU etiquette: waits while another job holds results/GPU_IN_USE.txt; writes its own marker while it runs.
# * End: results/$RUN_NAME/DONE.txt (the Windows notifier watches it). After 3 failed attempts it writes
#   FAILED.txt instead and stops retrying (delete FAILED.txt and restart the container to try again).
set -u
NAME="${RUN_NAME:-openlane_7models}"
RUN="/app/results/$NAME"
DONE="$RUN/DONE.txt"
FAILED="$RUN/FAILED.txt"
TRIES="$RUN/.attempts"
MARK=/app/results/GPU_IN_USE.txt
TAG="$NAME container"
mkdir -p "$RUN"

if [ -f "$DONE" ] || [ -f "$FAILED" ]; then
  echo "nothing to do: $(cat "$DONE" "$FAILED" 2>/dev/null | head -1)"
  exec sleep infinity
fi
while [ -f "$MARK" ] && ! grep -q "$TAG" "$MARK"; do
  echo "$(date '+%F %T') GPU held by another job: $(head -1 "$MARK"). Waiting 5 min."
  sleep 300
done
echo "GPU reserved by the $TAG (docker-compose.openlane.yml), started $(date '+%F %T'), expected several days. Removed when it ends." > "$MARK"

n=$(( $(cat "$TRIES" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$TRIES"
echo "$(date '+%F %T') $NAME attempt $n"

python -m tac_ufld run --config "configs/$NAME.yaml" --confirm --resume \
  && python -m tac_ufld robustness --runs "$RUN" --workers 2
rc=$?   # captured right away: after `if ...; fi` $? is always 0, which hid the kills (137 = SIGKILL, e.g. OOM)
if [ "$rc" -eq 0 ]; then
  echo "finished $(date '+%F %T'): report in results/$NAME/report/REPORT.md" > "$DONE"
  rm -f "$MARK" "$TRIES"
  exec sleep infinity
fi

echo "$(date '+%F %T') attempt $n failed (rc=$rc)"
if [ "$n" -ge 3 ]; then
  echo "failed 3 times, last rc=$rc at $(date '+%F %T'); see: docker logs tac-ufld-$NAME" > "$FAILED"
  rm -f "$MARK"
  exec sleep infinity
fi
exit "$rc"   # Docker restarts the container; --resume continues
