"""Build ``dist/tac_ufld_run.py``: the one-file runner for the supervisor (code packed inside).

    python scripts/supervisor/build_runner.py [--out dist/tac_ufld_run.py]

Packed: ``src/tac_ufld`` (without ``sim/``, the CARLA code), the experiment configs (without the CARLA ones),
``pyproject.toml`` and a dataset registry with ELAS, CULane, TuSimple and OpenLane enabled. Not packed: tests (except the synthetic-data builder),
results, datasets, checkpoints, CARLA scripts and docs. Rebuild after any code change.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).with_name("runner_template.py")
SKIP_PARTS = {"__pycache__", "sim", ".pytest_cache"}
SKIP_CONFIGS = ("carla",)

REGISTRY = """# Dataset registry of the bundle (2026-10-23): CULane is the experiment; the others are options (set
# enabled: true and uncomment the dataset in tac_ufld_run.py). Roots are relative to tac_ufld_code/.
datasets:
  culane:
    enabled: true
    root: "${CULANE_ROOT:-../datasets/CULane}"
    config: configs/culane.yaml                 # core plan; full: configs/cluster/culane_full.yaml
    smoke_config: configs/culane_smoke.yaml
    description: "CULane (Pan et al. 2018), 4 lanes, official lists"
  elas:
    enabled: false
    root: "${ELAS_ROOT:-../datasets/dataset_elas_v1}"
    config: configs/elas_pilot_v3.yaml
    smoke_config: configs/elas_smoke.yaml
    description: "ELAS (Berriel et al. 2017), 2 ego lanes, 30 fps, lane-change labels"
  openlane:
    enabled: false
    root: "${OPENLANE_ROOT:-../datasets/OpenLane}"
    config: configs/openlane.yaml
    smoke_config: configs/openlane_smoke.yaml
    description: "OpenLane v1 (2D lanes), 4 attribute slots"
  tusimple:
    enabled: false
    root: "${TUSIMPLE_ROOT:-../datasets/TUSimple}"
    config: configs/tusimple.yaml
    smoke_config: configs/tusimple_smoke.yaml
    description: "TuSimple lane benchmark, 4 lanes, 20-frame clips"
"""


def files() -> list[Path]:
    out = []
    for p in sorted((ROOT / "src" / "tac_ufld").rglob("*")):
        rel = p.relative_to(ROOT)
        if p.is_file() and not set(rel.parts) & SKIP_PARTS and p.suffix not in (".pyc",):
            out.append(rel)
    for p in sorted((ROOT / "configs").rglob("*.yaml")):
        rel = p.relative_to(ROOT)
        if not any(s in p.name for s in SKIP_CONFIGS) and p.name != "datasets.yaml":
            out.append(rel)
    out.append(Path("pyproject.toml"))
    # tools the runner calls (2026-10-23): dataset check, image cache, cluster jobs, the CULane plan
    extra = ["scripts/check_dataset.py", "scripts/cache_images.py", "scripts/cluster/culane.slurm",
             "scripts/cluster/setup_env.sh", "docs/PLAN_CULANE_2026-10-23.md", "docs/RUN_ON_CLUSTER.md",
             "docs/DATASET_DOWNLOAD_GUIDE.md"]
    out += [Path(e) for e in extra if (ROOT / e).exists()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "tac_ufld_run.py")
    args = ap.parse_args()
    buf = io.BytesIO()
    names = files()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in names:
            z.write(ROOT / rel, rel.as_posix())
        z.writestr("configs/datasets.yaml", REGISTRY)
        # generated CULane-format data for `smoke --synthetic` (the test fixture, packed as a tool)
        z.write(ROOT / "tests" / "dataset_fixtures.py", "tools/dataset_fixtures.py")
        z.writestr("tools/__init__.py", "")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = "?"
    info = f"{dt.date.today().isoformat()}, git {commit} + working tree, {len(names) + 1} files"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    lines = "\n".join(f'    "{b64[i:i + 100]}"' for i in range(0, len(b64), 100))
    text = TEMPLATE.read_text(encoding="utf-8").replace("__BUNDLE_INFO__", info).replace("__PAYLOAD__", lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({args.out.stat().st_size / 2**20:.2f} MB, {len(names) + 1} files packed: {info})")


if __name__ == "__main__":
    main()
