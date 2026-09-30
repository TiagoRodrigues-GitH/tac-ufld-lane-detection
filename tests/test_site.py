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


def test_site_adds_further_runs_with_a_head_to_head(tmp_path):
    main = _fake_run(tmp_path)
    lite = tmp_path / "lite_long"
    (lite / "report").mkdir(parents=True)
    rows = [{"seed": s, "variant": v, "split": "test", "protocol": "tuned", "input": "full", "lane_f1_iou50": f,
             "lane_f1_iou35": f, "pixel_f1": 0.3, "anchor_f1": 0.5, "jitter_px": 2.0, "lane_fp_iou50": 1,
             "lane_fn_iou50": 1} for s in (1, 2) for v, f in (("lite_baseline", 0.5), ("lite_v05", 0.7))]
    pd.DataFrame(rows).to_csv(lite / "all_results.csv", index=False)
    (lite / "config_resolved.yaml").write_text("description: longer lite run\ntrain: {epochs: 16}\n", encoding="utf-8")
    summary = tmp_path / "summary.md"
    summary.write_text("# s\n## Challenges\n- **small** data\n## Next steps\n1. full run\n", encoding="utf-8")
    text = build_site([main, lite], tmp_path / "site", "T", summary=summary).read_text(encoding="utf-8")
    assert "Head to head on the held-out roads" in text and "longer lite run" in text
    assert "Lite v0.5 warped (16 ep)" in text and "UFLD baseline (? ep)" in text
    # the one-screen overview comes first and takes the lite family from the longer run
    assert text.index('id="summary"') < text.index('id="results"')
    assert "<strong>small</strong> data" in text and "<ol><li>full run</li></ol>" in text


def test_choice_section_is_bilingual_and_keeps_model_names(tmp_path):
    from tac_ufld.site import _bi, _choice_section, _pt

    assert _pt("UFLD v0.3 +0.114 (0.875 × 0.761)") == "UFLD v0.3 +0,114 (0,875 × 0,761)"
    assert _bi("Sim", "Yes") == '<span lang="pt">Sim</span><span class="en" lang="en">Yes</span>'
    run = _fake_run(tmp_path)
    notes = tmp_path / "choice.md"
    notes.write_text("## Recomendação\n- usar **v0.3**\n## Recommendation\n- use **v0.3**\n## Notas por modelo\n"
                     "- `ufld_baseline`: referência || reference\n", encoding="utf-8")
    df = pd.read_csv(run / "all_results.csv")
    html = _choice_section(run, df, [], notes)
    assert 'id="choice"' in html and "Qual modelo comparar com a baseline" in html
    assert html.index('lang="pt"') < html.index('lang="en"')  # Portuguese first
    assert "usar <strong>v0.3</strong>" in html and "referência" in html
    assert "v0,3" not in html  # model names keep their dot


def test_portuguese_page_has_authors_no_english_left_and_prints_on_a4(tmp_path):
    from tac_ufld.site_i18n import translate_html

    run = _fake_run(tmp_path)
    notes = tmp_path / "notes.md"
    notes.write_text("- english **finding**\n", encoding="utf-8")
    notes.with_name("notes.pt.md").write_text("- resultado em **português**\n", encoding="utf-8")
    index = build_site([run], tmp_path / "pt", "T", notes=notes, lang="pt",
                       authors=["Tiago Rodrigues", "Eva Laussac", "Everton Gomede"], affiliation="UTFPR, Brazil")
    text = index.read_text(encoding="utf-8")
    assert '<html lang="pt-BR">' in text and "<title>Resultados TAC-UFLD</title>" in text
    assert "Autores:" in text and "Tiago Rodrigues, Eva Laussac, Everton Gomede · UTFPR, Brasil" in text
    assert "Onde a pesquisa está" in text and "Lane F1 em estradas nunca vistas" in text
    assert "resultado em <strong>português</strong>" in text and "english" not in text  # only the PT findings
    assert (tmp_path / "pt" / "untranslated.txt").read_text(encoding="utf-8") == ""
    assert "@media print" in text and "size: A4" in text
    # decimal comma in text, not in model names, code or markup
    out, missing = translate_html('<p>Lane F1 0.881 for UFLD v0.3 gated</p><code>x 0.5</code>'
                                  '<span class="en" lang="en">English half</span>')
    assert "0,881" in out and "v0.3" in out and "<code>x 0.5</code>" in out and "English half" not in out
    assert missing == ["Lane F1 0.881 for UFLD v0.3 gated"]
