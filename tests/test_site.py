"""The static results page builds from a run folder, is theme-aware, and
draws charts to scale from the run's numbers."""

from __future__ import annotations

import json
import re

import pandas as pd

from tac_ufld.site import bar_chart, build_site


def _fake_run(root):
    run = root / "fake_pilot"
    (run / "report").mkdir(parents=True)
    rows = []
    for seed in (1, 2):
        for v, f1 in (("ufld_baseline", 0.05), ("ufld_v02", 0.04), ("lite_baseline", 0.25), ("lite_v05", 0.40)):
            for split in ("test", "seen_test"):
                rows.append({"seed": seed, "variant": v, "split": split, "protocol": "tuned", "input": "full",
                             "lane_f1_iou50": f1 + 0.01 * seed, "lane_f1_iou35": f1 + 0.1, "pixel_f1": 0.3,
                             "anchor_f1": 0.5, "jitter_px": 2.5, "lane_fp_iou50": 100, "lane_fn_iou50": 100})
        for v in ("ufld_baseline", "lite_v05"):
            h = pd.DataFrame({"epoch": [1, 2, 3], "val_lane_f1_iou50": [0.2, 0.15, 0.1], "train_focal": [1, .5, .2]})
            (run / f"seed_{seed}").mkdir(exist_ok=True)
            h.to_csv(run / f"seed_{seed}" / f"history_{v}.csv", index=False)
    pd.DataFrame(rows).to_csv(run / "all_results.csv", index=False)
    (run / "environment.json").write_text(json.dumps({"gpu": "Test GPU"}), encoding="utf-8")
    (run / "run.log").write_text("10:00:00 INFO start\n12:22:00 INFO end\n", encoding="utf-8")
    pd.DataFrame([{"variant": "ufld_v02", "reference": "ufld_baseline", "metric": "lane_f1_iou50", "n_seeds": 2,
                   "mean_delta": -0.01, "ci95_low": -0.1, "ci95_high": 0.08, "wilcoxon_p": 0.5, "p_holm": 1.0,
                   "significant": False, "underpowered": True}]).to_csv(run / "report" / "paired_tests_test_tuned.csv",
                                                                          index=False)
    return run


def test_site_builds_from_a_run(tmp_path):
    run = _fake_run(tmp_path)
    notes = tmp_path / "notes.md"
    notes.write_text("# x\n- first **finding**\n- second `code`\n", encoding="utf-8")
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text("# r\n## Phase A\n- [done] built it\n- [later] embedded **last**\n", encoding="utf-8")
    index = build_site([run], tmp_path / "site", "Pilot Test", notes=notes, roadmap=roadmap)
    text = index.read_text(encoding="utf-8")
    assert "The models" in text and "Lite v0.5 warped" in text and "Difference" in text
    assert 'class="pill done"' in text and "<strong>last</strong>" in text
    assert text.startswith("<!doctype html>") and "<title>TAC-UFLD Pilot Results</title>" in text
    assert "color-scheme: light" in text and "body { background: var(--bg)" in text  # one explicit light look
    assert "<strong>finding</strong>" in text and "2 h 22 min" in text and "underpowered" in text
    assert "nan" not in re.sub(r"<[^>]+>", " ", text).lower().split()
    page = (tmp_path / "site" / "page.html").read_text(encoding="utf-8")
    assert "<html" not in page and page.startswith("<title>")
    assert (tmp_path / "site" / ".nojekyll").exists()


def test_bar_chart_is_to_scale():
    svg = bar_chart([{"variant": "ufld_baseline", "mean": 0.5, "seeds": [0.4, 0.6]}], "f1", width=760)
    width = float(re.search(r'<rect class="ufld" x="150" y="[\d.]+" width="([\d.]+)"', svg).group(1))
    assert abs(width - (760 - 150 - 60) * 0.5) < 0.2
