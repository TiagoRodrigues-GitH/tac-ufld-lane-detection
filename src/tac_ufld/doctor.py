"""Environment, hardware and dependency checks (``python -m tac_ufld doctor``)."""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys
from pathlib import Path

from tac_ufld.config import PROJECT_ROOT

OPTIONAL = {
    "onnx": "ONNX export", "onnxruntime": "ONNX Runtime inference", "onnxscript": "dynamo ONNX exporter",
    "onnxconverter_common": "ONNX FP16 conversion", "tensorrt": "TensorRT engines",
    "streamlit": "user interface", "carla": "CARLA demo (optional)",
}
REQUIRED = ("numpy", "pandas", "scipy", "cv2", "PIL", "matplotlib", "optuna", "yaml", "openpyxl", "tensorboard")


def _version(module: str) -> str | None:
    try:
        mod = importlib.import_module(module)
    except Exception:
        return None
    return str(getattr(mod, "__version__", "installed"))


def run_checks() -> dict:
    report: dict = {"python": sys.version.split()[0], "platform": platform.platform(),
                    "in_docker": Path("/.dockerenv").exists(),
                    "wsl": "microsoft" in platform.release().lower(), "problems": []}
    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report["gpu"] = {"name": props.name, "memory_gb": round(props.total_memory / 2**30, 2),
                             "compute_capability": f"{props.major}.{props.minor}",
                             "cudnn": torch.backends.cudnn.version()}
        else:
            report["problems"].append("no CUDA GPU visible: training will run on CPU (very slow)")
    except ImportError:
        report["torch"] = None
        report["problems"].append("PyTorch is not installed (see README: install torch first)")
    report["required"] = {m: _version(m) for m in REQUIRED}
    report["problems"] += [f"missing required package {m}" for m, v in report["required"].items() if v is None]
    report["optional"] = {m: {"version": _version(m), "used_for": use} for m, use in OPTIONAL.items()}
    ort = report["optional"]["onnxruntime"]["version"]
    if ort:
        import onnxruntime

        report["onnxruntime_providers"] = onnxruntime.get_available_providers()
    try:
        from tac_ufld.data.registry import DatasetRootError, load_registry

        datasets = {}
        for entry in load_registry().values():
            if not entry.enabled:
                datasets[entry.name] = "disabled"
                continue
            try:
                datasets[entry.name] = f"ok: {entry.resolve_root()}"
            except DatasetRootError as exc:
                datasets[entry.name] = f"ERROR: {exc}"
                report["problems"].append(str(exc))
        report["datasets"] = datasets
    except Exception as exc:  # the registry itself is broken
        report["datasets"] = {"error": str(exc)}
        report["problems"].append(f"dataset registry: {exc}")
    hub = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub" / "checkpoints"
    report["imagenet_weights_cached"] = any(hub.glob("resnet18-*.pth")) if hub.exists() else False
    results = PROJECT_ROOT / "results"
    target = results if results.exists() else PROJECT_ROOT
    report["disk_free_gb"] = round(shutil.disk_usage(target).free / 2**30, 1)
    if report["disk_free_gb"] < 20:
        report["problems"].append(f"only {report['disk_free_gb']} GB free: a full run writes ~10-20 GB")
    report["ok"] = not report["problems"]
    return report


def format_report(r: dict) -> str:
    lines = [f"Python {r['python']} on {r['platform']}" + (" (Docker)" if r["in_docker"] else "")
             + (" (WSL)" if r["wsl"] else ""),
             f"PyTorch {r.get('torch')} (CUDA build {r.get('torch_cuda_build')}), CUDA available: {r.get('cuda_available')}"]
    if "gpu" in r:
        g = r["gpu"]
        lines.append(f"GPU: {g['name']}, {g['memory_gb']} GB, compute {g['compute_capability']}, cuDNN {g['cudnn']}")
    lines.append("Required: " + ", ".join(f"{k} {v or 'MISSING'}" for k, v in r["required"].items()))
    lines.append("Optional: " + ", ".join(f"{k} {v['version'] or '-'}" for k, v in r["optional"].items()))
    if "onnxruntime_providers" in r:
        lines.append(f"ONNX Runtime providers: {r['onnxruntime_providers']}")
    lines.append("Datasets: " + ", ".join(f"{k}: {v}" for k, v in r.get("datasets", {}).items()))
    lines.append(f"ImageNet ResNet-18 weights cached: {r['imagenet_weights_cached']}; free disk: {r['disk_free_gb']} GB")
    lines.append("OK" if r["ok"] else "PROBLEMS:\n  - " + "\n  - ".join(r["problems"]))
    return "\n".join(lines)
