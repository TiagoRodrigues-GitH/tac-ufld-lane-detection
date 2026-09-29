"""Docker configuration is consistent with the package and never bakes data,
results or weights into the image."""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest
import yaml

from tac_ufld.config import PROJECT_ROOT


def test_dockerfile_references_existing_files_and_extras():
    text = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    for src in re.findall(r"^COPY\s+(.+?)\s+\S+$", text, flags=re.M):
        for item in src.split():
            if item != ".":
                assert (PROJECT_ROOT / item).exists(), item
    extras = re.search(r"ARG EXTRAS=(\S+)", text).group(1).split(",")
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for extra in extras:
        assert re.search(rf"^{extra} = \[", pyproject, flags=re.M), extra
    assert "datasets" not in text.split("COPY")[-1].split("\n")[0]


def test_dockerignore_excludes_data_results_weights_and_envs():
    ignored = set((PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").split())
    assert {"results", "datasets", ".venv", "*.pt", "*.pth", "*.onnx", ".git"} <= ignored


def test_compose_mounts_data_read_only_and_never_starts_the_full_run():
    compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert {"doctor", "tests", "smoke", "pilot", "ui"} <= set(services)
    common = compose["x-common"]
    data_mounts = [v for v in common["volumes"] if ":/data/" in v]
    assert data_mounts and all(v.endswith(":ro") for v in data_mounts)
    for name, svc in services.items():
        command = " ".join(svc.get("command", []))
        assert "configs/elas.yaml" not in command and "--confirm" not in command, name
    if (PROJECT_ROOT / ".git").exists():  # the placeholder is excluded from the image itself
        assert (PROJECT_ROOT / ".empty").is_dir()


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not installed")
def test_docker_compose_config_is_valid():
    res = subprocess.run(["docker", "compose", "config", "-q"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
