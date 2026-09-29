"""Dataset registry (``configs/datasets.yaml``): which datasets may be used and
where they live.

Rules:

* only datasets with ``enabled: true`` can be selected; a disabled dataset is
  never scanned - its root is not even expanded or checked;
* ``root`` accepts ``${VAR}`` and ``${VAR:-default}``; relative paths are
  relative to the project root;
* an enabled dataset whose root variable is unset or whose folder does not
  exist is an explicit error (never a silent fallback);
* ELAS is the only dataset enabled by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from tac_ufld.config import PROJECT_ROOT, ConfigError, expand_env
from tac_ufld.data import KNOWN_DATASETS

DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "datasets.yaml"
_ENTRY_KEYS = {"enabled", "root", "config", "smoke_config", "description"}


class DatasetRootError(ConfigError):
    """An enabled dataset's root is unset or missing."""


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    enabled: bool
    root: str | None
    config: str | None = None
    smoke_config: str | None = None
    description: str = ""

    def resolve_root(self) -> Path:
        """Expand and check the root. Call only for enabled datasets."""
        if not self.root:
            raise DatasetRootError(f"dataset '{self.name}' is enabled but has no root in the registry")
        try:
            text = expand_env(self.root, f"datasets.{self.name}.root")
        except ConfigError as exc:
            raise DatasetRootError(f"dataset '{self.name}': {exc}. Set it or disable the dataset.") from exc
        path = Path(text)
        path = path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        if not path.is_dir():
            raise DatasetRootError(f"dataset '{self.name}' is enabled but its root does not exist: {path}")
        return path

    def config_path(self, smoke: bool = False) -> Path:
        rel = self.smoke_config if smoke and self.smoke_config else self.config
        if not rel:
            raise ConfigError(f"dataset '{self.name}' has no {'smoke_' if smoke else ''}config in the registry")
        path = Path(rel)
        return path if path.is_absolute() else PROJECT_ROOT / path


def load_registry(path: str | Path | None = None) -> dict[str, DatasetEntry]:
    path = Path(path) if path else DEFAULT_REGISTRY
    if not path.exists():
        raise ConfigError(f"dataset registry not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if set(raw) != {"datasets"}:
        raise ConfigError(f"{path}: expected a single top-level 'datasets' mapping, got {sorted(raw)}")
    entries = {}
    for name, body in (raw["datasets"] or {}).items():
        key = str(name).lower()
        if key not in KNOWN_DATASETS:
            raise ConfigError(f"{path}: unknown dataset '{name}'; supported: {list(KNOWN_DATASETS)}")
        body = body or {}
        if not isinstance(body, dict):
            raise ConfigError(f"{path}: datasets.{name} must be a mapping")
        unknown = set(body) - _ENTRY_KEYS
        if unknown:
            raise ConfigError(f"{path}: unknown key(s) {sorted(unknown)} in datasets.{name}")
        enabled = body.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigError(f"{path}: datasets.{name}.enabled must be true or false")
        entries[key] = DatasetEntry(key, enabled, body.get("root"), body.get("config"),
                                    body.get("smoke_config"), body.get("description", ""))
    return entries


def require_enabled(registry: dict[str, DatasetEntry], name: str) -> DatasetEntry:
    """The registry entry of ``name`` if it exists and is enabled (the root is
    not checked here)."""
    key = name.lower()
    if key not in registry:
        raise ConfigError(f"dataset '{name}' is not in the registry (available: {sorted(registry)})")
    if not registry[key].enabled:
        raise ConfigError(f"dataset '{name}' is disabled in the registry; set "
                          f"'datasets.{key}.enabled: true' to use it")
    return registry[key]


def select_datasets(registry: dict[str, DatasetEntry], names: list[str] | None = None) -> list[DatasetEntry]:
    """``names=None`` -> every enabled dataset. Explicit names must exist and
    be enabled. Roots of the selected datasets are checked up front, so a
    multi-dataset run fails before any training starts."""
    if names is None:
        chosen = [e for e in registry.values() if e.enabled]
        if not chosen:
            raise ConfigError("no dataset is enabled in the registry")
    else:
        chosen = [require_enabled(registry, name) for name in names]
    for entry in chosen:
        entry.resolve_root()
    return chosen
