"""Seeding, logging and small I/O helpers."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch. ``deterministic`` trades speed for
    bit-exact reruns (cuDNN autotuning off, deterministic kernels)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init: derive python/numpy seeds from torch's per-worker seed."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        root.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve()
                   for h in root.handlers):
            handler = logging.FileHandler(log_file, encoding="utf-8")
            handler.setFormatter(fmt)
            root.addHandler(handler)
    return logging.getLogger("tac_ufld")


def environment_info() -> dict[str, str]:
    info = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "cuda_available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
