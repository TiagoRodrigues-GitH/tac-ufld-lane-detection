"""Build the OpenLane-V input cache straight from the OpenLane image archives, without extracting them.

OpenLane-V needs only the cropped, resized band of each frame (``scripts/build_openlane_v_cache.py``: top 480 rows
removed, 1920x800 -> 640x384, ~50 KB per frame). Extracting the 20 archives first writes ~103 GB of full-size
images and then reads them back; on a DRAM-less SATA SSD that took about an hour per archive (2026-10-05). This
script reads each archive as a stream (from the downloaded .tar if it is on disk, otherwise from the Hugging Face
mirror over HTTP), converts only the frames of OpenLane-V segments and writes ~6 GB in total.

    python scripts/stream_openlane_v_cache.py D:\\datasets\\OpenLane                # 4 archives at a time
    python scripts/stream_openlane_v_cache.py D:\\datasets\\OpenLane --workers 2

* Resumable: a finished archive leaves ``<cache>/<part>.vcache`` and is skipped; inside an archive, frames
  already in the cache are skipped. A dropped HTTP connection is resumed at the same byte (Range request).
* Segments already extracted under ``images/`` are converted too (``build_openlane_v_cache.py``).
* A downloaded ``.tar`` is deleted once its frames are cached (``--keep-archives`` to keep it).
* Exit code 0 only if every frame listed by OpenLane-V is in the cache.
"""

from __future__ import annotations

import argparse
import io
import logging
import pickle
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_openlane_v_cache import CROP_TOP, SOURCE_SIZE, SPLITS, convert_segment  # noqa: E402
from download_openlane import HF_FILE, IMAGE_PARTS  # noqa: E402

LOGGER = logging.getLogger("stream_openlane_v_cache")
RETRIES = 30


class ResumableHTTPStream(io.RawIOBase):
    """Sequential reader over an HTTP file that reconnects at the current byte when the connection drops."""

    def __init__(self, url: str) -> None:
        self.url, self.pos, self.response, self.total = url, 0, None, None

    def readable(self) -> bool:
        return True

    def _open(self) -> None:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-"})
        self.response = urllib.request.urlopen(req, timeout=120)
        if self.pos and self.response.status != 206:
            raise OSError("server ignored the Range header; cannot resume")
        if self.total is None:  # "Content-Range: bytes 0-N/TOTAL"
            content_range = self.response.headers.get("Content-Range", "")
            self.total = int(content_range.rsplit("/", 1)[1]) if "/" in content_range else None

    def readinto(self, buffer) -> int:
        for attempt in range(RETRIES):
            try:
                if self.response is None:
                    self._open()
                n = self.response.readinto(buffer)
                if n == 0 and self.total is not None and self.pos < self.total:
                    raise OSError("connection closed before the end of the file")
                self.pos += n
                return n
            except OSError as exc:  # timeouts, resets, early close: reconnect at self.pos
                LOGGER.warning("%s: %s at byte %d, reconnecting (%d)", self.url.rsplit("/", 1)[-1], exc, self.pos,
                               attempt + 1)
                self.response = None
                time.sleep(min(60, 5 * (attempt + 1)))
        raise OSError(f"{self.url}: gave up after {RETRIES} reconnections")


def needed_segments(root: Path) -> dict[str, set[str]]:
    out = {}
    for split in SPLITS:
        with (root / "OpenLane-V" / "list" / f"datalist_{split}.pickle").open("rb") as f:
            out[split] = {name.split("/")[0] for name in pickle.load(f)}
    return out


def convert_bytes(data: bytes, dst: Path, size: tuple[int, int]) -> None:
    with Image.open(io.BytesIO(data)) as img:
        img.draft("RGB", (SOURCE_SIZE[0] // 2, SOURCE_SIZE[1] // 2))
        scale = img.size[1] / SOURCE_SIZE[1]
        band = img.convert("RGB").crop((0, round(CROP_TOP * scale), img.size[0], img.size[1]))
        out = band.resize(size, Image.BILINEAR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    out.save(tmp, "JPEG", quality=95)
    tmp.replace(dst)


def cache_part(part: str, archives: Path, out_root: Path, needed: dict[str, set[str]],
               size: tuple[int, int], keep_archive: bool) -> str:
    marker = archives / f"{part}.vcache"
    if marker.exists():
        return f"{part}: already cached"
    if (archives / f"{part}.done").exists():  # extracted by download_openlane.py: cache_extracted converts it
        return f"{part}: extracted earlier"
    split = "training" if part.startswith("images_training") else "validation"
    local = archives / f"{part}.tar"
    origin = "the downloaded archive" if local.exists() else "the mirror (streamed)"
    source = (local.open("rb") if local.exists()
              else io.BufferedReader(ResumableHTTPStream(HF_FILE.format(name=f"{part}.tar")), buffer_size=8 << 20))
    written = skipped = 0
    with source, tarfile.open(fileobj=source, mode="r|") as tar:
        for member in tar:  # images_<split>_<k>/<segment>/<timestamp>.jpg
            pieces = member.name.split("/")
            if not member.isfile() or len(pieces) < 3 or pieces[-2] not in needed[split]:
                continue
            dst = out_root / split / pieces[-2] / pieces[-1]
            if dst.exists():
                skipped += 1
                continue
            convert_bytes(tar.extractfile(member).read(), dst, size)
            written += 1
    marker.write_text(f"{written} {skipped}", encoding="ascii")
    if local.exists() and not keep_archive:
        local.unlink()
    return f"{part}: {written} frames cached ({skipped} already there) from {origin}"


def cache_extracted(root: Path, out_root: Path, needed: dict[str, set[str]], size: tuple[int, int]) -> str:
    """Segments that an earlier download already extracted under images/<split>/."""
    done = 0
    for split in SPLITS:
        for segment in sorted(needed[split]):
            src = root / "images" / split / segment
            if src.is_dir():
                convert_segment(src, out_root / split / segment, size)
                done += 1
    return f"extracted segments: {done} converted"


def missing_frames(root: Path, out_root: Path) -> int:
    missing = 0
    for split in SPLITS:
        with (root / "OpenLane-V" / "list" / f"datalist_{split}.pickle").open("rb") as f:
            for name in pickle.load(f):
                missing += not (out_root / split / f"{name}.jpg").exists()
    return missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="OpenLane folder with OpenLane-V/ (and images/ if extracted)")
    ap.add_argument("--archives", type=Path, help="where .tar parts are (default: <root>/../_downloads/OpenLane)")
    ap.add_argument("--size", default="640x384")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep-archives", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    w, h = (int(v) for v in a.size.lower().split("x"))
    root = a.root.resolve()
    archives = (a.archives or root.parent / "_downloads" / "OpenLane").resolve()
    out_root = root / "OpenLane-V" / f"images_{w}x{h}"
    needed = needed_segments(root)
    failed = 0
    with ThreadPoolExecutor(a.workers + 1) as pool:
        jobs = {pool.submit(cache_extracted, root, out_root, needed, (w, h)): "extracted segments"}
        jobs.update({pool.submit(cache_part, p, archives, out_root, needed, (w, h), a.keep_archives): p
                     for p in IMAGE_PARTS})
        for job in as_completed(jobs):
            try:
                LOGGER.info(job.result())
            except Exception as exc:  # a failed part is retried by the next run
                failed += 1
                LOGGER.error("%s failed: %s", jobs[job], exc)
    missing = missing_frames(root, out_root)
    LOGGER.info("OpenLane-V frames missing from the cache: %d", missing)
    return 0 if failed == 0 and missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
