"""
config/loader.py — YAML config loader with CLI override support.

Loads base.yaml, optionally deep-merges a dict of overrides, and applies
any --set key=value strings from the CLI.  The result is a Config object
whose attributes mirror the YAML hierarchy (e.g. cfg.vad.hangover_ms).

Usage (programmatic):
    from config.loader import load_config
    cfg = load_config(path="config/base.yaml", set_args=["vad.hangover_ms=300"])

The module-level `cfg` singleton is imported by src/config.py on startup.
"""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Any
import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "base.yaml"


def _deep_merge(base: dict, overrides: dict) -> dict:
    """
    Recursively merge `overrides` into a deep copy of `base`.
    Nested dicts are merged rather than replaced wholesale.
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_dotted_override(cfg_dict: dict, dotted_key: str, raw_value: str) -> None:
    """
    Write a scalar into `cfg_dict` at the path described by `dotted_key`.

    Intermediate dicts are created automatically.  `raw_value` is
    auto-coerced: "true"/"false" → bool, integer string → int,
    float string → float, anything else → str.

    Example:
        _apply_dotted_override(d, "vad.hangover_ms", "300")
        # equivalent to: d["vad"]["hangover_ms"] = 300
    """
    keys = dotted_key.split(".")
    node = cfg_dict
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    leaf_key = keys[-1]
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


def load_config(path=None, overrides=None, set_args=None):
    """
    Load and return a Config object.

    Args:
        path:      Path to a YAML file.  Defaults to config/base.yaml.
        overrides: Dict to deep-merge on top of the file contents.
        set_args:  List of "key=value" strings (from --set CLI args).
                   Applied after `overrides`.

    Returns:
        Config — attribute-style access to every key in the YAML.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    if set_args:
        for arg in set_args:
            if "=" not in arg:
                raise ValueError(
                    f"--set argument must be key=value, got: '{arg}'")
            key, _, value = arg.partition("=")
            _apply_dotted_override(raw, key.strip(), value.strip())
    return Config(raw)


class Config:
    """
    Attribute-style wrapper around a nested dict.

    Nested dicts become child Config objects so callers can write
    cfg.vad.hangover_ms instead of cfg["vad"]["hangover_ms"].
    """

    def __init__(self, data: dict) -> None:
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Return `key` from the underlying dict, or `default` if absent."""
        return self._data.get(key, default)

    def as_dict(self) -> dict:
        """Return a deep copy of the raw dict (useful for serialization)."""
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


# Module-level singleton — imported by src/config.py
cfg: Config = load_config()
