"""Can ELAS support ego-lane and lane-change analysis? Run ``tac_ufld.ego`` on ELAS lanes, annotated or predicted.

Feeds the ego-left/ego-right boundaries of every frame (in order, at 29.97 FPS) to ``LaneChangeDetector`` and
scores its per-frame decisions against the ELAS ``laneChange`` label (-1 left, +1 right).

* without ``--checkpoint``: the ground-truth lanes, i.e. the upper bound of any detector built on a lane model;
* with ``--checkpoint``: the lanes a trained model predicts, frame by frame through ``StreamingLaneDetector``
  (the deployment path). Use scenes the model was NOT trained on: the lane-change scenes outside the 10-scene
  configs are GRI_S02, ROD_S01 and VV_S02.

    python scripts/ego_lane_check.py [--scenes GRI_S02 ROD_S01 VV_S02] [--checkpoint <model.pt>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_ufld.data.elas import discover_scenes, parse_groundtruth, parse_scene_config  # noqa: E402
from tac_ufld.ego import EgoLaneTracker, LaneChangeParams, event_metrics  # noqa: E402


def _predicted_lanes(loaded, paths, cfg):
    """Frame id -> predicted lanes (original pixels), restricted to the scene ROI like the evaluation."""
    import cv2

    from tac_ufld.inference.streaming import StreamingLaneDetector

    det = StreamingLaneDetector(loaded, valid_y_range=(float(cfg.roi_y), float(cfg.roi_y + cfg.roi_h)))
    out = {}
    for fid in range(cfg.frame_start, cfg.frame_end + 1):
        path = paths.images / f"lane_{fid}.png"
        if not path.exists():
            continue
        bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        res = det.infer_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), stream_id=cfg.scene, frame_index=fid,
                              sequence_id=cfg.scene)
        out[fid] = list(res.lanes)
    return out


def run_scene(paths, params: LaneChangeParams, loaded=None) -> tuple[np.ndarray, np.ndarray, list[float]]:
    cfg = parse_scene_config(paths.config)
    ann = parse_groundtruth(paths.groundtruth, cfg)
    predicted = _predicted_lanes(loaded, paths, cfg) if loaded is not None else None
    tracker = EgoLaneTracker(cfg.frame_w / 2.0, cfg.roi_y + cfg.roi_h, cfg.roi_y, params)
    pred, gt, center_err = [], [], []
    for fid in range(cfg.frame_start, cfg.frame_end + 1):
        a = ann.get(fid)
        if predicted is not None:
            lanes = predicted.get(fid, [None, None])
        else:
            lanes = list(a.lanes) if a is not None else [None, None]
        s, decision = tracker.update(lanes)
        if a is not None and np.isfinite(s.left_x) and np.isfinite(s.right_x):
            label_center = a.attrs.get("laneCenter", float("nan"))
            if np.isfinite(label_center) and np.isfinite(s.center_x):
                center_err.append(abs(s.center_x - label_center))
        pred.append(decision)
        label = a.attrs.get("laneChange", 0.0) if a is not None else 0.0
        gt.append(int(label) if np.isfinite(label) else 0)
    return np.asarray(pred), np.asarray(gt), center_err


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../../datasets/dataset_elas_v1")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--checkpoint", default=None, help="score a model's predicted lanes instead of the annotation")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    index = discover_scenes(Path(args.root))
    scenes = args.scenes or sorted(index)
    loaded = None
    if args.checkpoint:
        import torch

        from tac_ufld.inference.loading import load_model

        device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
        loaded = load_model(args.checkpoint, device)
        print(f"predicted lanes of {loaded.variant} ({args.checkpoint})")
    else:
        print("ground-truth lanes (upper bound)")
    params = LaneChangeParams()
    all_pred, all_gt, all_err = [], [], []
    print(f"{'scene':8s} {'frames':>6s} {'events':>6s} {'found':>5s} {'dir ok':>6s} {'false':>5s} {'delay s':>7s}")
    for name in scenes:
        pred, gt, err = run_scene(index[name], params, loaded)
        m = event_metrics(pred, gt, params.fps)
        print(f"{name:8s} {len(gt):6d} {m['gt_events']:6d} {m['detected']:5d} {m['direction_correct']:6d} "
              f"{m['false_alarms']:5d} {m['median_delay_s']:7.2f}")
        all_pred.append(np.append(pred, 0))   # a 0 between scenes keeps events from merging
        all_gt.append(np.append(gt, 0))
        all_err += err
    m = event_metrics(np.concatenate(all_pred), np.concatenate(all_gt), params.fps)
    print("\nall scenes:", {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})
    print(f"lane-centre |error| vs the ELAS laneCenter label: median {np.median(all_err):.2f} px "
          f"over {len(all_err)} frames with both boundaries")


if __name__ == "__main__":
    main()
