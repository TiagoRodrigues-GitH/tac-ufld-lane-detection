#!/usr/bin/env bash
# Publish the static results page to the `gh-pages` branch (GitHub Pages).
#
#   bash scripts/publish_pages.sh                       # default runs, findings and title (below)
#   bash scripts/publish_pages.sh --runs results/<run> [results/<run2> ...] --notes <findings.md> --title "..."
#
# Builds site/ with `python -m tac_ufld site` (arguments are passed through),
# copies it into a temporary worktree of the gh-pages branch (created as an
# orphan branch the first time), commits and pushes. The working branch is
# not touched. Then, once: GitHub -> Settings -> Pages -> Build and deployment
# -> "Deploy from a branch" -> gh-pages / (root).
# Note: on a PRIVATE repository GitHub Pages needs a paid plan (GitHub Pro is
# free with the Student Developer Pack), and the published page is public.
set -euo pipefail

PY="${PYTHON:-python}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
if [ $# -eq 0 ]; then
  set -- --runs results/elas_pilot_v2 results/elas_lite_long --notes docs/PILOT_V2_FINDINGS.md          --title "TAC-UFLD: temporal lane detection pilots"
fi

"$PY" -m tac_ufld site --out site "$@"
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
  git commit -m "Publish results page from $(git -C "$ROOT" rev-parse --short HEAD)" || echo "nothing to publish"
  git push origin gh-pages
)
echo "Published. Enable Pages once: Settings -> Pages -> Deploy from a branch -> gh-pages / (root)."
