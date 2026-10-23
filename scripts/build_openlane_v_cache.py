"""Build the OpenLane-V input images: every frame of the OpenLane-V segments, cropped and resized once.

RVLD's protocol (Jin et al., ICCV 2023; ``config.py``: ``crop_size = 480``, ``height, width = 384, 640``)
drops the top 480 rows of the 1920x1280 frame (sky) and feeds the remaining 1920x800 band at 640x384. Doing
that once here keeps JPEG decoding off the training loop: a 640x384 JPEG decodes in ~2 ms against ~40 ms
for the original. History frames are frames of the same segment, so every frame of a listed segment is
cached, not only the annotated ones.

    python scripts/build_openlane_v_cache.py D:\\datasets\\OpenLane             # 640x384, 6 processes
    python scripts/build_openlane_v_cache.py D:\\datasets\\OpenLane --workers 8

Output: ``<root>/OpenLane-V/images_<W>x<H>/{training,validation}/<segment>/<timestamp>.jpg`` (quality 95).
Rerunnable: existing files are skipped, so it can run while the image archives are still downloading.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

LOGGER = logging.getLogger("build_openlane_v_cache")
CROP_TOP = 480           # rows of the 1920x1280 frame removed by RVLD
SOURCE_SIZE = (1920, 1280)
SPLITS = ("training", "validation")


def listed_segments(root: Path, split: str) -> list[str]:
    with (root / "OpenLane-V" / "list" / f"datalist_{split}.pickle").open("rb") as f:
        return sorted({name.split("/")[0] for name in pickle.load(f)})


def convert(src: Path, dst: Path, size: tuple[int, int]) -> bool:
    """Crop the top rows and resize; False if the source is unreadable (e.g. still being extracted)."""
    try:
        with Image.open(src) as img:
            img.draft("RGB", (SOURCE_SIZE[0] // 2, SOURCE_SIZE[1] // 2))  # JPEG DCT scaling: decode at 960x640
            scale = img.size[1] / SOURCE_SIZE[1]
            band = img.convert("RGB").crop((0, round(CROP_TOP * scale), img.size[0], img.size[1]))
            out = band.resize(size, Image.BILINEAR)
    except OSError:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    out.save(tmp, "JPEG", quality=95)
    tmp.replace(dst)  # a crash never leaves a half-written image behind
    return True


def convert_segment(src_dir: Path, dst_dir: Path, size: tuple[int, int]) -> tuple[int, int, int]:
    """(converted, skipped, failed) for one segment folder."""
    done = skipped = failed = 0
    for src in sorted(src_dir.glob("*.jpg")):
        dst = dst_dir / src.name
        if dst.exists():
            skipped += 1
        elif convert(src, dst, size):
            done += 1
        else:
            failed += 1
    return done, skipped, failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="OpenLane folder with images/ and OpenLane-V/")
    ap.add_argument("--size", default="640x384", help="WxH of the cached images (the network input size)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    w, h = (int(v) for v in a.size.lower().split("x"))
    out_root = a.root / "OpenLane-V" / f"images_{w}x{h}"
    jobs, missing = [], 0
    for split in SPLITS:
        for segment in listed_segments(a.root, split):
            src = a.root / "images" / split / segment
            if src.is_dir():
                jobs.append((src, out_root / split / segment))
            else:
                missing += 1
    LOGGER.info("%d segments with images, %d not downloaded yet", len(jobs), missing)
    totals = [0, 0, 0]
    with ProcessPoolExecutor(a.workers) as pool:
        for i, counts in enumerate(pool.map(convert_segment, *zip(*jobs), [(w, h)] * len(jobs)), 1):
            totals = [t + c for t, c in zip(totals, counts)]
            if i % 50 == 0 or i == len(jobs):
                LOGGER.info("%d/%d segments: %d converted, %d already there, %d unreadable", i, len(jobs), *totals)
    if missing or totals[2]:
        LOGGER.warning("incomplete: %d segments missing, %d unreadable frames; run again when the download ends",
                       missing, totals[2])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
