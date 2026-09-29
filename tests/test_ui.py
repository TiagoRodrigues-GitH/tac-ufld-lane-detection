"""The Streamlit interface starts without training, renders with and without
checkpoints, and never imports the training code."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from tac_ufld.config import PROJECT_ROOT
from tests.test_streaming import _cfg, _checkpoint

pytest.importorskip("streamlit")
APP = PROJECT_ROOT / "src" / "tac_ufld" / "ui" / "app.py"


def test_ui_imports_no_training_code():
    code = ("import sys; import tac_ufld.ui.core, tac_ufld.inference.streaming; "
            "bad = sorted(m for m in sys.modules if m.startswith(('tac_ufld.training', 'tac_ufld.experiment', "
            "'tac_ufld.losses', 'optuna'))); print(bad); sys.exit(1 if bad else 0)")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")})
    assert res.returncode == 0, res.stdout + res.stderr


def _app_test(results: Path):
    from streamlit.testing.v1 import AppTest

    os.environ["TAC_UFLD_RESULTS"] = str(results)
    try:
        at = AppTest.from_file(str(APP), default_timeout=120)
        at.run()
    finally:
        os.environ.pop("TAC_UFLD_RESULTS", None)
    return at


def test_app_renders_without_results(tmp_path):
    at = _app_test(tmp_path / "empty")
    assert not at.exception
    assert any("No checkpoints" in w.value for w in at.sidebar.warning)
    assert len(at.tabs) == 6


def test_app_renders_with_checkpoints_and_conventions(tmp_path):
    ck = tmp_path / "results" / "run_a" / "seed_1" / "checkpoints"
    ck.mkdir(parents=True)
    _checkpoint(ck, "ufld_baseline", _cfg())
    ck_b = tmp_path / "results" / "run_b" / "seed_1" / "checkpoints"
    ck_b.mkdir(parents=True)
    _checkpoint(ck_b, "lite_v05", _cfg(**{"data.griding_num": 20}))
    at = _app_test(tmp_path / "results")
    assert not at.exception
    assert len(at.sidebar.multiselect[0].value) == 2
    assert any("griding_num" in w.value for w in at.warning)  # different output conventions are flagged


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_streamlit_server_starts_headless(tmp_path):
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "tac_ufld", "ui", "--headless", "--port", str(port),
                             "--results", str(tmp_path)], cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")})
    try:
        deadline, body = time.time() + 90, ""
        while time.time() < deadline and proc.poll() is None:
            try:
                body = urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2).read().decode()
                if body == "ok":
                    break
            except OSError:
                time.sleep(1)
        assert body == "ok"
    finally:
        proc.terminate()
        proc.wait(timeout=30)
