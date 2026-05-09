"""
config/loader.py — YAML config loader with CLI override support.

Usage:
    from config.loader import cfg

    python main.py --config config/base.yaml --set vad.hangover_ms=300
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "base.yaml"


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base. Returns a new dict."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_dotted_override(cfg_dict: dict, dotted_key: str, raw_value: str) -> None:
    """Set cfg_dict[a][b][c] = cast(raw_value) for a dotted key like 'vad.hangover_ms'."""
    keys = dotted_key.split(".")
    node = cfg_dict
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]

    leaf_key = keys[-1]
    # Attempt to preserve types: bool > int > float > str
    lower = raw_value.lower()
    if lower in ("true", "false"):
        node[leaf_key] = lower == "true"
    else:
        try:
            node[leaf_key] = int(raw_value)
        except ValueError:
            try:
                node[leaf_key] = float(raw_value)
            except ValueError:
                node[leaf_key] = raw_value


def load_config(
    path: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
    set_args: list[str] | None = None,
) -> "Config":
    """
    Load YAML config, apply dict overrides, then apply dotted-key --set overrides.

    Args:
        path:      Path to YAML file. Defaults to config/base.yaml.
        overrides: Dict of nested overrides deep-merged on top of the file.
        set_args:  List of 'key.subkey=value' strings (from --set CLI flag).

    Returns:
        Config — attribute-access wrapper around the merged dict.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        raw = _deep_merge(raw, overrides)

    if set_args:
        for arg in set_args:
            if "=" not in arg:
                raise ValueError(f"--set argument must be key=value, got: '{arg}'")
            key, _, value = arg.partition("=")
            _apply_dotted_override(raw, key.strip(), value.strip())

    return Config(raw)


class Config:
    """Thin attribute-access wrapper around a nested dict."""

    def __init__(self, data: dict) -> None:
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def as_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


# Module-level singleton — replaced by load_config() calls from main.py / enroll.py
cfg: Config = load_config()
