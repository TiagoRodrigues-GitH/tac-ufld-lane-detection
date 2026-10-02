"""Writes data.image_cache: every image under data.root decoded at the network size, so training and evaluation
skip decoding large sources each epoch (OpenLane: 1920x1280 -> 480x320, ~2 GB instead of 23.6 GB).

The pixels are produced exactly as tac_ufld.data.transforms.load_frame produces them (JPEG draft decoding when
data.jpeg_draft, then PIL bilinear resize), then stored as JPEG quality 95 without chroma subsampling: the only
difference from reading the original is that re-encoding (well below the variation of the photometric augmentation).
Existing files are skipped, so the script can be stopped and resumed.

    $env:OPENLANE_ROOT = "...\\datasets\\OpenLane"
    python scripts/cache_images.py --config configs/openlane.yaml [--workers 4]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def decode(src: Path, img_w: int, img_h: int, draft: bool) -> Image.Image:
    """Same steps as load_frame, before the float conversion."""
    with Image.open(src) as img:
        if draft and img.format == "JPEG":
            img.draft("RGB", (img_w, img_h))
        return img.convert("RGB").resize((img_w, img_h), resample=Image.BILINEAR)


def cache_one(args: tuple[str, str, int, int, bool]) -> int:
    src, dst, img_w, img_h, draft = args
    out = Path(dst)
    if out.exists():
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    decode(Path(src), img_w, img_h, draft).save(tmp, format="JPEG", quality=95, subsampling=0)
    os.replace(tmp, out)  # never leaves a half-written file under the final name
    return 1


def main() -> int:
    from tac_ufld.config import load_config
    from tac_ufld.data.dataset import _absolute

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cfg = load_config(a.config)
    d = cfg.data
    if not d.image_cache:
        raise SystemExit("data.image_cache is not set in the config")
    root, cache = cfg.data_root(), _absolute(d.image_cache, "data.image_cache")
    jobs = []
    for dirpath, _, files in os.walk(root):
        if Path(dirpath).resolve().is_relative_to(cache):
            continue
        for name in files:
            src = Path(dirpath) / name
            if src.suffix.lower() in EXTENSIONS:
                dst = cache / src.relative_to(root).with_suffix(".jpg")
                jobs.append((str(src), str(dst), d.img_w, d.img_h, d.jpeg_draft))
    print(f"{len(jobs)} images under {root} -> {cache} ({d.img_w}x{d.img_h})", flush=True)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(a.workers) as pool:
        for k, written in enumerate(pool.map(cache_one, jobs, chunksize=64), 1):
            done += written
            if k % 5000 == 0:
                print(f"  {k}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    size = sum(f.stat().st_size for f in cache.rglob("*.jpg")) / 2**30
    print(f"{done} new files, cache {size:.2f} GB, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
