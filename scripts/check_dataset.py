"""Check that a downloaded lane dataset is complete and laid out as TAC-UFLD expects.

Plain Python 3.9+ is enough (no project installation, no GPU). If Pillow is
installed, the image size of a sample is checked too.

    python scripts/check_dataset.py culane   D:\\datasets\\CULane
    python scripts/check_dataset.py tusimple D:\\datasets\\TUSimple
    python scripts/check_dataset.py openlane D:\\datasets\\OpenLane
    python scripts/check_dataset.py elas     D:\\datasets\\dataset_elas_v1

The script only reads files; it never changes, moves or deletes anything.
It prints one line per check:

    [ OK ]  found and complete
    [WARN]  usable, but look at it (e.g. a count differs from the published one)
    [MISS]  missing: the line says which download provides it

and a final verdict: READY (exit code 0) or NOT READY (exit code 1).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:  # optional
    from PIL import Image
except ImportError:  # pragma: no cover - depends on the machine
    Image = None

SAMPLES = 20  # annotation/image pairs checked per split


class Report:
    def __init__(self, dataset: str, root: Path) -> None:
        self.dataset, self.root = dataset, root
        self.missing_items = 0
        self.warnings = 0
        print(f"\nChecking {dataset} in {root}\n")

    def ok(self, msg: str) -> None:
        print(f"  [ OK ]  {msg}")

    def warn(self, msg: str) -> None:
        self.warnings += 1
        print(f"  [WARN]  {msg}")

    def miss(self, msg: str) -> None:
        self.missing_items += 1
        print(f"  [MISS]  {msg}")

    def info(self, msg: str) -> None:
        print(f"          {msg}")

    def count(self, what: str, found: int, expected: int | None) -> None:
        if expected is None:
            self.ok(f"{what}: {found:,}")
        elif found == expected:
            self.ok(f"{what}: {found:,} (published: {expected:,})")
        elif found == 0:
            self.miss(f"{what}: 0 (published: {expected:,})")
        else:
            self.warn(f"{what}: {found:,}, published: {expected:,} (incomplete download or another version?)")

    def pairs(self, what: str, checked: int, bad: list[str]) -> None:
        if checked == 0:
            self.miss(f"{what}: nothing to check")
        elif not bad:
            self.ok(f"{what}: {checked} samples, every file present")
        else:
            self.miss(f"{what}: {len(bad)} of {checked} samples incomplete, e.g. {bad[0]}")

    def finish(self) -> int:
        print()
        if self.missing_items:
            print(f"NOT READY: {self.missing_items} item(s) marked [MISS] above. "
                  f"See docs/DATASET_DOWNLOAD_GUIDE.md.")
            return 1
        extra = f" ({self.warnings} warning(s) to look at)" if self.warnings else ""
        print(f"READY: {self.dataset} is complete and in the expected layout{extra}.")
        return 0


# ------------------------------------------------------------------ helpers


def count_lines(path: Path) -> int:
    with open(path, "rb") as fh:
        return sum(1 for line in fh if line.strip())


def count_files(folder: Path, suffix: str) -> int:
    n = 0
    for _, _, files in os.walk(folder):
        n += sum(1 for f in files if f.lower().endswith(suffix))
    return n


def spread(items: list, k: int = SAMPLES) -> list:
    """k items evenly spread over the list (deterministic)."""
    if len(items) <= k:
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def find_base(root: Path, marker: str, depth: int = 2) -> Path | None:
    """``root`` or a folder up to ``depth`` levels below it that contains ``marker``."""
    level = [root]
    for _ in range(depth + 1):
        for d in level:
            if (d / marker).exists():
                return d
        level = [c for d in level for c in sorted(d.iterdir()) if c.is_dir()] if level else []
    return None


def image_size(path: Path, report: Report, expected: tuple[int, int]) -> None:
    if Image is None:
        report.info("(install Pillow to also check the image size)")
        return
    with Image.open(path) as img:
        size = img.size
    if size == expected:
        report.ok(f"image size {size[0]}x{size[1]}")
    else:
        report.warn(f"image size {size[0]}x{size[1]}, expected {expected[0]}x{expected[1]} ({path.name})")


def moved_root(report: Report, root: Path, base: Path) -> None:
    if base != root:
        report.warn(f"the data sits one level down: use {base} as the dataset root")


# ------------------------------------------------------------------ CULane

CULANE_LISTS = {"train_gt.txt": 88880, "val_gt.txt": 9675, "test.txt": 34680}  # CULane website
CULANE_DRIVERS = {"driver_23_30frame": "train/val", "driver_161_90frame": "train/val",
                  "driver_182_30frame": "train/val", "driver_37_30frame": "test",
                  "driver_100_30frame": "test", "driver_193_90frame": "test"}


def check_culane(root: Path) -> Report:
    r = Report("CULane", root)
    base = find_base(root, "list/train_gt.txt")
    if base is None:
        r.miss("list/train_gt.txt  -> extract list.tar.gz into the dataset root")
        return r
    moved_root(r, root, base)
    for name, expected in CULANE_LISTS.items():
        path = base / "list" / name
        if path.exists():
            r.count(f"list/{name}", count_lines(path), expected)
        else:
            r.miss(f"list/{name}  -> extract list.tar.gz into the dataset root")
    categories = sorted((base / "list" / "test_split").glob("test*_*.txt"))
    if len(categories) == 9:
        r.ok(f"list/test_split/: 9 test categories ({', '.join(p.stem.split('_', 1)[1] for p in categories)})")
    else:
        r.warn(f"list/test_split/: {len(categories)} category lists, 9 expected (per-condition results need them)")
    for driver, part in CULANE_DRIVERS.items():
        if (base / driver).is_dir():
            r.ok(f"{driver}/ ({part}): {count_files(base / driver, '.jpg'):,} images")
        else:
            r.miss(f"{driver}/ ({part})  -> extract {driver}.tar.gz into the dataset root")
    if (base / "laneseg_label_w16").is_dir():
        r.ok("laneseg_label_w16/ (lane labels used to order the lanes)")
    else:
        r.miss("laneseg_label_w16/  -> extract laneseg_label_w16.tar.gz into the dataset root")
    for name in CULANE_LISTS:
        path = base / "list" / name
        if not path.exists():
            continue
        lines = [ln.split() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        bad, history = [], 0
        for parts in spread(lines):
            img = base / parts[0].lstrip("/")
            need = [img, img.with_suffix(".lines.txt")]
            if len(parts) > 1:
                need.append(base / parts[1].lstrip("/"))
            bad += [str(p.relative_to(base)) for p in need if not p.exists()]
            try:  # history frame 90 video frames earlier (configs/culane.yaml temporal_step)
                prev = img.with_name(f"{int(img.stem) - 90:05d}.jpg")
                history += prev.exists()
            except ValueError:
                pass
        r.pairs(f"{name}: image + .lines.txt{' + seg label' if name != 'test.txt' else ''}",
                len(spread(lines)), bad)
        r.info(f"history frame (t - 90 video frames) available for {history}/{len(spread(lines))} samples "
               f"(clip starts have none; the code then repeats the nearest newer frame)")
        if name == "train_gt.txt" and lines:
            image_size(base / lines[0][0].lstrip("/"), r, (1640, 590))
    r.info("Remember: annotations_new.tar.gz must be extracted OVER the train/val drivers "
           "(it replaces the old, incorrect .lines.txt files).")
    return r


# ---------------------------------------------------------------- TuSimple

TUSIMPLE_TRAIN = 3626   # annotated training clips (TuSimple benchmark)
TUSIMPLE_TEST = 2782    # annotated test clips


def _tusimple_pairs(r: Report, label: Path, split_dir: Path, what: str) -> None:
    lines = [ln for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]
    bad, checked = [], 0
    for ln in spread(lines):
        checked += 1
        try:
            item = json.loads(ln)
            raw = item["raw_file"]
            assert "lanes" in item and "h_samples" in item
        except (ValueError, KeyError, AssertionError):
            bad.append(f"{label.name}: unreadable line")
            continue
        img = next((b / raw for b in (split_dir, label.parent, split_dir.parent) if (b / raw).exists()), None)
        if img is None:
            bad.append(raw)
            continue
        for k in (18, 16):  # history frames of configs/tusimple.yaml (temporal_step 2)
            if not img.with_name(f"{k}.jpg").exists():
                bad.append(str(img.with_name(f"{k}.jpg")))
    r.pairs(f"{what}: annotated frame 20 + history frames 18, 16", checked, bad)


def check_tusimple(root: Path) -> Report:
    r = Report("TuSimple", root)
    base = find_base(root, "train_set")
    if base is None:
        r.miss("train_set/  -> unzip the Kaggle download; the dataset root is the folder that holds "
               "train_set/, test_set/ and test_label.json (called TUSimple/ in the Kaggle archive)")
        return r
    moved_root(r, root, base)
    labels = sorted((base / "train_set").glob("label_data_*.json"))
    names = {p.name for p in labels}
    for name in ("label_data_0313.json", "label_data_0531.json", "label_data_0601.json"):
        if name not in names:
            r.miss(f"train_set/{name}")
    r.count("annotated training clips (label_data_*.json)", sum(count_lines(p) for p in labels), TUSIMPLE_TRAIN)
    test_label = next((p for p in (base / "test_label.json", base / "test_set" / "test_label.json") if p.exists()),
                      None)
    if test_label is None:
        r.miss("test_label.json (test ground truth)  -> in the Kaggle archive it sits next to test_set/")
    else:
        r.count(f"annotated test clips ({test_label.relative_to(base)})", count_lines(test_label), TUSIMPLE_TEST)
    for folder in ("train_set/clips", "test_set/clips"):
        if (base / folder).is_dir():
            r.ok(f"{folder}/: {count_files(base / folder, '.jpg'):,} images")
        else:
            r.miss(f"{folder}/")
    for p in labels:
        _tusimple_pairs(r, p, base / "train_set", p.name)
    if test_label is not None:
        _tusimple_pairs(r, test_label, base / "test_set", "test_label.json")
    first = next(iter(sorted((base / "train_set" / "clips").rglob("20.jpg"))), None) \
        if (base / "train_set" / "clips").is_dir() else None
    if first:
        image_size(first, r, (1280, 720))
    return r


# ---------------------------------------------------------------- OpenLane

OPENLANE_SEGMENTS = {"lane3d_1000": {"training": 798, "validation": 202},   # OpenLane data README
                     "lane3d_300": {"training": 240, "validation": 60}}


def check_openlane(root: Path) -> Report:
    r = Report("OpenLane (v1, 2D lanes)", root)
    base = find_base(root, "images", depth=1)
    if base is None:
        r.miss("images/  -> extract images_training_*.tar and images_validation_*.tar "
               "(or images.tar) into the dataset root")
        return r
    moved_root(r, root, base)
    ann_name = next((n for n in OPENLANE_SEGMENTS if (base / n).is_dir()), None)
    if ann_name is None:
        r.miss("lane3d_1000/  -> extract lane3d_1000_training.tar and lane3d_1000_validation_test.tar "
               "into the dataset root (or lane3d_300.tar for the small subset)")
        return r
    r.ok(f"annotations: {ann_name}/")
    for split, expected in OPENLANE_SEGMENTS[ann_name].items():
        ann_dir, img_dir = base / ann_name / split, base / "images" / split
        if not ann_dir.is_dir():
            r.miss(f"{ann_name}/{split}/")
            continue
        segments = sorted(d for d in ann_dir.iterdir() if d.is_dir())
        r.count(f"{ann_name}/{split}: segments", len(segments), expected)
        r.ok(f"{ann_name}/{split}: {count_files(ann_dir, '.json'):,} annotation files")
        if not img_dir.is_dir():
            r.miss(f"images/{split}/")
            continue
        img_segments = {d.name for d in img_dir.iterdir() if d.is_dir()}
        lost = [s.name for s in segments if s.name not in img_segments]
        if lost:
            r.miss(f"images/{split}: {len(lost)} annotated segments have no image folder, e.g. {lost[0]} "
                   f"(an images_{split}_*.tar part is missing?)")
        else:
            r.ok(f"images/{split}: every annotated segment has its image folder")
        files = [f for s in spread(segments) for f in sorted(s.glob("*.json"))[:1]]
        bad = []
        for js in files:
            try:
                item = json.loads(js.read_text(encoding="utf-8"))
                assert "lane_lines" in item
            except (ValueError, AssertionError):
                bad.append(f"{js.name}: unreadable or no lane_lines")
                continue
            img = img_dir / js.parent.name / (js.stem + ".jpg")
            if not img.exists():
                bad.append(str(img.relative_to(base)))
        r.pairs(f"{split}: annotation + image", len(files), bad)
        if split == "training" and files and not bad:
            image_size(img_dir / files[0].parent.name / (files[0].stem + ".jpg"), r, (1920, 1280))
    cases = sorted(d.name for d in (base / ann_name / "test").iterdir() if d.is_dir()) \
        if (base / ann_name / "test").is_dir() else []
    if cases:
        r.ok(f"{ann_name}/test/ scenario subsets: {', '.join(cases)}")
    else:
        r.warn(f"{ann_name}/test/ scenario subsets not found (optional: only per-scenario results need them)")
    return r


# -------------------------------------------------------------------- ELAS

ELAS_HELD_OUT = ("BR_S02", "VIX_S05", "VV_S03")  # test scenes of configs/elas*.yaml


def check_elas(root: Path) -> Report:
    r = Report("ELAS", root)
    scenes = {}
    for d, dirs, files in os.walk(root):
        if "config.xml" in files and "groundtruth.xml" in files:
            scenes[Path(d).name] = Path(d)
            dirs[:] = []
    if not scenes:
        r.miss("no scene folder with config.xml + groundtruth.xml below the root")
        return r
    r.ok(f"{len(scenes)} scenes: {', '.join(sorted(scenes))}")
    for name in ELAS_HELD_OUT:
        if name not in scenes:
            r.miss(f"scene {name} (a held-out test scene of the ELAS protocol)")
    for name, d in sorted(scenes.items()):
        images = d / "images" / "images"
        n_img = count_files(images, ".png") + count_files(images, ".jpg") if images.is_dir() else 0
        n_gt = d.joinpath("groundtruth.xml").read_text(encoding="utf-8", errors="replace").count("<frame ")
        if n_img == 0:
            hint = " (images.zip is there: unzip it to use this scene)" if (d / "images.zip").exists() else ""
            if name in ELAS_HELD_OUT:
                r.miss(f"{name}: no images in {images.relative_to(root)}{hint}")
            else:  # the configs list the scenes they use; an unused scene may stay zipped
                r.warn(f"{name}: no images in {images.relative_to(root)}{hint}; "
                       f"only a problem if a config lists this scene")
        elif n_img < n_gt:
            r.warn(f"{name}: {n_img:,} images for {n_gt:,} annotated frames")
        else:
            r.ok(f"{name}: {n_img:,} images, {n_gt:,} annotated frames")
    return r


CHECKS = {"culane": check_culane, "tusimple": check_tusimple, "openlane": check_openlane, "elas": check_elas}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=sorted(CHECKS))
    p.add_argument("root", help="folder where the dataset was extracted")
    args = p.parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"NOT READY: {root} is not a folder")
        return 1
    return CHECKS[args.dataset](root).finish()


if __name__ == "__main__":
    sys.exit(main())
