#!/usr/bin/env bash
# Publish the static results page to the `gh-pages` branch (GitHub Pages).
#
#   bash scripts/publish_pages.sh [results/elas_pilot] ["TAC-UFLD pilot results"]
#
# Builds site/ with `python -m tac_ufld site`, copies it into a temporary
# worktree of the gh-pages branch (created as an orphan branch the first
# time), commits and pushes. The working branch is not touched.
# Then, once: GitHub -> Settings -> Pages -> Build and deployment ->
# "Deploy from a branch" -> gh-pages / (root).
# Note: on a PRIVATE repository GitHub Pages needs a paid plan (GitHub Pro is
# free with the Student Developer Pack), and the published page is public.
set -euo pipefail

RUN="${1:-results/elas_pilot}"
TITLE="${2:-TAC-UFLD pilot results}"
PY="${PYTHON:-python}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

"$PY" -m tac_ufld site --runs "$RUN" --title "$TITLE" --out site
WT="$(mktemp -d)"
cleanup() { git worktree remove --force "$WT" 2>/dev/null || true; }
trap cleanup EXIT

if git ls-remote --exit-code --heads origin gh-pages >/dev/null 2>&1; then
  git fetch origin gh-pages
  git worktree add "$WT" origin/gh-pages --detach
  (cd "$WT" && git checkout -B gh-pages)
else
  git worktree add --detach "$WT"
  (cd "$WT" && git checkout --orphan gh-pages && git rm -rf . >/dev/null 2>&1 || true)
fi

find "$WT" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -r site/. "$WT"/
rm -f "$WT/page.html"   # embedding variant, not needed on Pages
(
  cd "$WT"
  git add -A
  git commit -m "Publish results page from $(git -C "$ROOT" rev-parse --short HEAD) ($RUN)" || echo "nothing to publish"
  git push origin gh-pages
)
echo "Published. Enable Pages once: Settings -> Pages -> Deploy from a branch -> gh-pages / (root)."
