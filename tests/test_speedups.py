"""Opt-in speed-ups for large datasets: frame scoring in worker processes (evaluation.workers) and images read
from a cache decoded at the network size (data.image_cache). Both must leave every result unchanged."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tac_ufld.config import PROJECT_ROOT, DataConfig, load_config
from tac_ufld.data.dataset import image_cache_paths
from tac_ufld.data.openlane import OpenLaneAdapter
from tac_ufld.data.targets import encode_targets, make_row_anchors
from tac_ufld.data.transforms import load_frame
from tac_ufld.evaluation.evaluator import Evaluator
from tac_ufld.evaluation.predictor import Predictions
from tac_ufld.postprocess import PostprocessParams
from tests.dataset_fixtures import make_openlane

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from cache_images import cache_one  # noqa: E402


@pytest.fixture(scope="module")
def openlane_case(tmp_path_factory):
    root = make_openlane(tmp_path_factory.mktemp("openlane"), frames_per_segment=8)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPENLANE_ROOT", str(root))
        cfg = load_config(PROJECT_ROOT / "configs" / "openlane.yaml")
    d = cfg.data
    anchors = make_row_anchors(d.img_h, d.num_row_anchors, d.row_anchor_range)
    records = OpenLaneAdapter(root).official_splits()["train"]
    rng = np.random.default_rng(0)
    exist, x = [], []
    for rec in records:  # imperfect predictions: the targets with noise and some missed anchors
        t = encode_targets(rec, d.img_w, d.img_h, anchors, d.griding_num)
        exist.append(np.clip(t.exist * rng.uniform(0.3, 1.0, t.exist.shape), 0, 1).astype(np.float32))
        x.append((t.x + rng.normal(0, 6, t.x.shape)).astype(np.float32))
    preds = Predictions(np.stack(exist), np.stack(x), np.full(len(records), np.nan, np.float32))
    yield cfg, anchors, records, preds


def _same(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for k, v in a.items():
        assert v == b[k] or (np.isnan(v) and np.isnan(b[k])), k


def test_parallel_scoring_gives_the_same_results(openlane_case):
    cfg, anchors, records, preds = openlane_case
    params = PostprocessParams(threshold=0.5, min_points=2)
    cfg.evaluation.workers = 1
    serial = Evaluator(cfg, anchors).evaluate(preds, records, params)
    cfg.evaluation.workers = 2
    parallel = Evaluator(cfg, anchors).evaluate(preds, records, params)
    _same(serial.metrics, parallel.metrics)
    assert serial.metrics["native_openlane_tp_iou50"] > 0  # the all-lanes metric was really exercised
    assert serial.per_frame.equals(parallel.per_frame)
    for a, b in zip(serial.pred_lanes, parallel.pred_lanes):
        assert all((p is None and q is None) or np.array_equal(p, q) for p, q in zip(a, b))
    lane_only = Evaluator(cfg, anchors).evaluate(preds, records, params, lane_only=True)
    cfg.evaluation.workers = 1
    _same(lane_only.metrics, Evaluator(cfg, anchors).evaluate(preds, records, params, lane_only=True).metrics)


def test_image_cache_maps_paths_and_falls_back(tmp_path):
    root, cache = tmp_path / "data", tmp_path / "cache"
    (root / "seq").mkdir(parents=True)
    (cache / "seq").mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (64, 48), (10, 20, 30)).save(root / "seq" / name)
    Image.new("RGB", (32, 24)).save(cache / "seq" / "a.jpg")
    assert image_cache_paths(DataConfig(root=str(root))) is None  # off by default
    cached = image_cache_paths(DataConfig(root=str(root), image_cache=str(cache)))
    assert Path(cached(str(root / "seq" / "a.jpg"))) == cache / "seq" / "a.jpg"
    assert Path(cached(str(root / "seq" / "b.jpg"))) == root / "seq" / "b.jpg"  # no copy: original
    assert cached.stats() == (1, 1)


def test_cached_image_matches_the_decoded_original(tmp_path):
    rng = np.random.default_rng(1)
    src = tmp_path / "frame.jpg"
    Image.fromarray(rng.integers(0, 255, (1280, 1920, 3), dtype=np.uint8)).save(src, quality=95)
    dst = tmp_path / "cache" / "frame.jpg"
    assert cache_one((str(src), str(dst), 480, 320, True)) == 1
    assert cache_one((str(src), str(dst), 480, 320, True)) == 0  # already there
    original = load_frame(str(src), 480, 320, draft=True)
    from_cache = load_frame(str(dst), 480, 320, draft=True)
    assert from_cache.shape == original.shape
    # only the JPEG re-encoding differs (random noise is the worst case for JPEG)
    assert float((from_cache - original).abs().mean()) < 0.03
