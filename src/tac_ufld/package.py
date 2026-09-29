"""Supervisor hand-off ZIP (``python -m tac_ufld package``).

Contents: every file git tracks or would track (untracked files that are not
ignored), i.e. source, configs, tests, docs, Docker files, UI, deployment
scripts, the static results site and the supervisor's reference code, plus -
optionally - the lightweight reports of finished runs (CSV / Markdown / PNG /
JSON, never checkpoints). Datasets, checkpoints, exported models, virtual
environments and caches are excluded by construction and double-checked.
A ``MANIFEST.txt`` lists every file with its SHA-256 and the git commit.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import subprocess
import zipfile
from pathlib import Path

from tac_ufld import __version__
from tac_ufld.config import PROJECT_ROOT

EXCLUDED_SUFFIXES = {".pt", ".pth", ".onnx", ".engine", ".plan", ".ckpt", ".db", ".pkl", ".mp4", ".avi",
                     ".zip", ".tfevents", ".prof", ".log"}
EXCLUDED_PARTS = {".venv", "venv", "env", "__pycache__", ".pytest_cache", "datasets", "results", "dist",
                  ".git", "checkpoints", "tensorboard", "build", ".ipynb_checkpoints"}
RESULT_PATTERNS = ("config_resolved.yaml", "environment.json", "all_results.csv", "report/*.md", "report/*.csv",
                   "report/*.png", "report/*.json", "data/split_summary.csv", "data/split_report.json",
                   "data/label_geometry_check.csv", "hpo/*.json", "seed_*/history_*.csv", "seed_*/results.csv",
                   "seed_*/postprocess/*_tuned.json", "seed_*/plots/*.png",
                   "deploy/*/numerical_checks.json", "deploy/*/deployment.json",   # export checks
                   "*.md", "*.json",                                                # benchmark reports
                   "index.html", "assets/*.jpg")                                    # offline results page
MAX_RESULT_FILE_MB = 5


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def source_files() -> list[Path]:
    names = set(_git("ls-files", "-z").split("\0")) | set(_git("ls-files", "--others", "--exclude-standard",
                                                               "-z").split("\0"))
    files = []
    for name in sorted(n for n in names if n):
        path = PROJECT_ROOT / name
        if path.is_file() and not _excluded(Path(name)):
            files.append(path)
    return files


def _excluded(rel: Path) -> bool:
    return bool(set(rel.parts) & EXCLUDED_PARTS) or rel.suffix.lower() in EXCLUDED_SUFFIXES


def result_files(run: Path) -> list[Path]:
    out: dict[Path, None] = {}  # ordered and unique: several patterns can match one file
    for pattern in RESULT_PATTERNS:
        for p in sorted(run.glob(pattern)):
            if p.is_file() and p.stat().st_size <= MAX_RESULT_FILE_MB * 2**20 and p.suffix.lower() not in EXCLUDED_SUFFIXES:
                out[p] = None
    return list(out)


def build_package(out_dir: Path, include_results: list[Path]) -> tuple[Path, str]:
    commit = _git("rev-parse", "--short", "HEAD").strip()
    dirty = bool(_git("status", "--porcelain").strip())
    stamp = dt.date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"tac-ufld-handoff-v{__version__}-{stamp}.zip"
    root = f"tac-ufld-lane-detection-v{__version__}"
    entries: list[tuple[Path, str]] = [(p, p.relative_to(PROJECT_ROOT).as_posix()) for p in source_files()]
    for run in include_results:
        run = run if run.is_absolute() else PROJECT_ROOT / run
        for p in result_files(run):
            entries.append((p, f"pilot_results/{run.name}/{p.relative_to(run).as_posix()}"))
    arcnames = [a for _, a in entries]
    duplicates = sorted({a for a in arcnames if arcnames.count(a) > 1})
    if duplicates:
        raise RuntimeError(f"duplicate archive entries: {duplicates[:5]}")
    manifest = [f"TAC-UFLD hand-off package v{__version__}, git {commit}{' + uncommitted changes' if dirty else ''}, "
                f"built {stamp}", ""]
    total = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path, arc in entries:
            data = path.read_bytes()
            total += len(data)
            zf.writestr(f"{root}/{arc}", data)
            manifest.append(f"{hashlib.sha256(data).hexdigest()}  {len(data):>10}  {arc}")
        zf.writestr(f"{root}/MANIFEST.txt", "\n".join(manifest) + "\n")
    names = [a for _, a in entries]
    leaks = [n for n in names if _excluded(Path(n)) and not n.startswith("pilot_results/")]
    report = (f"{len(entries)} files, {total / 2**20:.1f} MB uncompressed, {zip_path.stat().st_size / 2**20:.1f} MB "
              f"zipped; git {commit}{' (dirty)' if dirty else ''}; "
              f"{sum(n.startswith('pilot_results/') for n in names)} result files from {len(include_results)} run(s)")
    if leaks:
        raise RuntimeError(f"excluded files would be packaged: {leaks[:5]}")
    return zip_path, report
