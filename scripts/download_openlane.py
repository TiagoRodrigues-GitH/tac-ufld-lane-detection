"""Download and lay out the full OpenLane v1 image set (20 parts, ~103 GB) next to its lane annotations.

Resumable: each archive goes through download -> integrity check -> extraction -> segment move -> deletion,
and a marker file per archive (``<name>.done``) lets a new run skip the finished ones. Peak extra disk use is
one ~5.7 GB archive per worker.

    python scripts/download_openlane.py D:\\datasets\\OpenLane                 # all missing parts, 3 at a time
    python scripts/download_openlane.py D:\\datasets\\OpenLane --annotations   # also extract every annotation

Result (as docs/DATASET_DOWNLOAD_GUIDE.md expects): ``images/{training,validation}/segment-*/*.jpg`` and
``lane3d_1000/{training,validation,test}/...``. A part already extracted as ``images_<split>_<k>/`` by an
older download is moved into place as well.

Source: the official Google Drive folder answers "too many users have downloaded this file" for days at a time,
so the image parts come from a public Hugging Face mirror of the same archives (identical sizes; every
download is checked against the SHA-256 the mirror publishes). Drive (``gdown``) is the fallback. Needs the
Windows or GNU ``tar``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LOGGER = logging.getLogger("download_openlane")

HF_REPO = "AlayaNeW/OpenDriveLab___OpenLane"  # mirror of the OpenDataLab copy, files under raw/
HF_TREE = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main/raw"
HF_FILE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/raw/{{name}}"
CHUNK = 8 << 20
RETRIES = 20

# Google Drive file ids of the official OpenLane folder (18upnDfB-VVuQf3GPiv_JQn1-BUOcAotk), split_tar/ parts.
IMAGE_PARTS = {
    "images_training_0": "1g14qx6-DQ28f6PZGMnXkpFGo4T5yF0tA",
    "images_training_1": "1k33TOn5pSza7AiR-nmk0LsVKGoZDlRD7",
    "images_training_2": "1E0L0TBoF3D0mE5z2dKrZ_nziOC_6kyuE",
    "images_training_3": "14WRXgHz1-3Lk0aayXhWZQX8rvsiNWHUc",
    "images_training_4": "1jEE-R1ShnKKF09SU13DSpd_EDVNXp172",
    "images_training_5": "1Od8dPvq_AKfyuK3DRt3ahrsMNVS6bKeS",
    "images_training_6": "11vrg-6xaDs2sw2RPFJssrDTIlQavVUKS",
    "images_training_7": "1U2PpLe_fctpQga29HigWi8Tm8yXTCUot",
    "images_training_8": "12v3V4OL_vm9Fq-SiabKRUw9lZjJ21x8D",
    "images_training_9": "1-OW6UYZLLxYq-DILMJ6uQRjp9cvCE6xh",
    "images_training_10": "13nz0KhxNhrfhGhgBzerElejif_2Nojhy",
    "images_training_11": "1NAFcB7KGCq8o_5w136hgm1Mq4fvHw3oN",
    "images_training_12": "1GBUl6NWZfScQy_5-hsbvYrfuW9uljvZB",
    "images_training_13": "1G2xNgbENrXCTLC5RvjKlsblV2iT4OXEO",
    "images_training_14": "1VhlfsJMLfmRbgsjlgKO95ZdUZ6KBuFs3",
    "images_training_15": "1sLzjI0aA0xqets_n1bqv43zozR6jnzyU",
    "images_validation_0": "1Ro6hJd7j836y2jPIDo5jVJk40VwXJBTq",
    "images_validation_1": "1A8FPq-rsXJgall-EC0Z3GR8PMi75fukm",
    "images_validation_2": "1vXXPvFNeG_c39EAgW5D39wqcd9Cte7RF",
    "images_validation_3": "10B47sXSd-gnLHKUcqeOPACy3bY9eBVrk",
}
ANNOTATIONS = {"lane3d_1000_training": "1jJ_7hEC9xpUVwZbQf7LdOmQR5udDJ-d7",
               "lane3d_1000_validation_test": "1KeplaxI9VqUEIh-Toyw9jYDVShJebJZE"}


def split_of(part: str) -> str:
    return "training" if part.startswith("images_training") else "validation"


def ensure_real_dir(path: Path) -> None:
    """images/<split> may be a junction to a single part (older layout): replace it by a real folder."""
    if path.is_junction() or path.is_symlink():
        path.rmdir()  # removes the link only, never the target
    path.mkdir(parents=True, exist_ok=True)


def move_segments(extracted: Path, dest: Path) -> int:
    if not extracted.exists():
        return 0
    moved = 0
    for seg in sorted(extracted.iterdir()):
        target = dest / seg.name
        if target.exists():
            continue
        seg.rename(target)  # same volume: a rename, no copy
        moved += 1
    if not any(extracted.iterdir()):
        extracted.rmdir()
    return moved


def mirror_index() -> dict[str, tuple[int, str]]:
    """name -> (size, sha256) of every archive on the mirror."""
    with urllib.request.urlopen(HF_TREE, timeout=60) as r:
        return {Path(f["path"]).name: (f["size"], f["lfs"]["oid"]) for f in json.load(r) if f.get("lfs")}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def download_mirror(archive: Path, size: int, digest: str) -> None:
    """HTTP download with resume (Range) from where a previous attempt stopped, then a SHA-256 check."""
    partial = archive.with_suffix(".tar.part")
    for _ in range(RETRIES):  # the CDN sometimes closes the connection early: continue from the last byte
        have = partial.stat().st_size if partial.exists() else 0
        if have >= size:
            break
        req = urllib.request.Request(HF_FILE.format(name=archive.name), headers={"Range": f"bytes={have}-"})
        with urllib.request.urlopen(req, timeout=120) as r, partial.open("ab") as out:
            if have and r.status != 206:  # server ignored the range: start over
                out.truncate(0)
            shutil.copyfileobj(r, out, CHUNK)
    if partial.stat().st_size != size:
        raise RuntimeError(f"{archive.name}: incomplete after {RETRIES} attempts; the next run resumes it")
    if sha256(partial) != digest:
        partial.unlink()
        raise RuntimeError(f"{archive.name}: SHA-256 differs from the mirror's; the next run downloads it again")
    partial.rename(archive)


def download_drive(file_id: str, archive: Path) -> None:
    import gdown

    out = gdown.download(id=file_id, output=str(archive), quiet=True, resume=True)
    if out is None:
        raise RuntimeError(f"gdown returned nothing for {archive.name} (Drive quota or link problem)")
    if not tar_ok(archive):  # Drive gives no checksum: list the archive instead (the mirror's SHA-256 covers it)
        archive.unlink()
        raise RuntimeError(f"{archive.name} from Drive is incomplete or corrupt; the next run downloads it again")


def download(file_id: str, archive: Path, index: dict[str, tuple[int, str]]) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.name in index:
        try:
            download_mirror(archive, *index[archive.name])
            return
        except OSError as exc:  # network error: try Drive before giving up
            LOGGER.warning("%s: mirror failed (%s); trying Google Drive", archive.name, exc)
    download_drive(file_id, archive)


def tar_ok(archive: Path) -> bool:
    return subprocess.run(["tar", "-tf", str(archive)], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def fetch_part(cache: Path, part: str, file_id: str, index: dict[str, tuple[int, str]]) -> None:
    """Stage 1 (parallel): make sure the archive is on disk unless the part is already extracted."""
    if (cache / f"{part}.done").exists() or (cache / f"{part}.tar.extracted").exists():
        return
    archive = cache / f"{part}.tar"
    if not archive.exists():  # a .tar appears only when complete (mirror: after the SHA-256 check)
        download(file_id, archive, index)


def install_part(root: Path, cache: Path, part: str) -> str:
    """Stage 2 (one at a time): extract, move the segments into images/<split>/, delete the archive."""
    done = cache / f"{part}.done"
    if done.exists():
        return f"{part}: already done"
    archive = cache / f"{part}.tar"
    extracted = cache / f"{part}.tar.extracted"  # same marker name as the first download (2026-10-01)
    if not extracted.exists():  # an interrupted extraction is redone from the archive
        subprocess.run(["tar", "-xf", str(archive), "-C", str(root)], check=True)
        extracted.write_text("ok", encoding="ascii")
    archive.unlink(missing_ok=True)
    moved = move_segments(root / part, root / "images" / split_of(part))
    done.write_text(str(moved), encoding="ascii")
    return f"{part}: {moved} segments into images/{split_of(part)}"


def extract_annotations(root: Path, cache: Path, index: dict[str, tuple[int, str]]) -> str:
    for name, file_id in ANNOTATIONS.items():
        done = cache / f"{name}.done"
        if done.exists():
            continue
        archive = cache / f"{name}.tar"
        if not (archive.exists() and tar_ok(archive)):
            download(file_id, archive, index)
        target = root / "lane3d_1000"
        target.mkdir(exist_ok=True)
        subprocess.run(["tar", "-xf", str(archive), "-C", str(target)], check=True)
        done.write_text("ok", encoding="ascii")
    return "annotations: lane3d_1000/ complete (archives kept in the cache)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="OpenLane folder (images/ and lane3d_1000/ go inside)")
    ap.add_argument("--cache", type=Path, help="where archives are downloaded (default: <root>/../_downloads/OpenLane)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--annotations", action="store_true", help="also extract the full annotation archives")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    root = a.root.resolve()
    cache = (a.cache or root.parent / "_downloads" / "OpenLane").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    for split in ("training", "validation"):
        ensure_real_dir(root / "images" / split)
    index = mirror_index()
    failed = 0
    # Two stages: archives download in parallel, and each finished archive is queued for the single extraction
    # thread (several tar processes writing thousands of files at once make a SATA SSD or an HDD crawl).
    with ThreadPoolExecutor(a.workers) as downloads, ThreadPoolExecutor(1) as extraction:
        installs = {}
        if a.annotations:
            installs[extraction.submit(extract_annotations, root, cache, index)] = "annotations"
        fetches = {downloads.submit(fetch_part, cache, p, i, index): p for p, i in IMAGE_PARTS.items()}
        for job in as_completed(fetches):
            part = fetches[job]
            try:
                job.result()
            except Exception as exc:  # one failed part must not stop the others; a rerun retries it
                failed += 1
                LOGGER.error("%s failed: %s", part, exc)
                continue
            installs[extraction.submit(install_part, root, cache, part)] = part
        for job in as_completed(installs):
            try:
                LOGGER.info(job.result())
            except Exception as exc:
                failed += 1
                LOGGER.error("%s failed: %s", installs[job], exc)
    for split in ("training", "validation"):
        n = sum(1 for _ in (root / "images" / split).iterdir())
        LOGGER.info("images/%s: %d segments (official: %d)", split, n, 798 if split == "training" else 202)
    LOGGER.info("free disk: %.0f GB", shutil.disk_usage(root).free / 1e9)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
