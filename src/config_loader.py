"""YAML config loader with CLI override support.

Reads the settings tree from config/base.yaml. Pipeline code does not use this
module directly -- it reads the flat constants in config.py, which this module
feeds.

Usage:
    python main.py --config config/base.yaml --set vad.hangover_ms=300
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# config/ holds only settings data; this loader lives with the source.
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "base.yaml"


class Config:
    """Thin attribute-access wrapper around a nested dict."""

    def __init__(self, data: dict) -> None:
        self._data = data
        for key, value in data.items():
            setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def get(self, key: str, default: Any = None) -> Any:
        """Return a top-level key, or ``default`` if it is absent."""
        return self._data.get(key, default)

    def as_dict(self) -> dict:
        """Return a deep copy of the underlying dict."""
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` into ``base``, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _coerce(raw_value: str) -> bool | int | float | str:
    """Parse a CLI string into the narrowest type it fits: bool, int, float, str."""
    lowered = raw_value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(raw_value)
        except ValueError:
            continue
    return raw_value


def _apply_dotted_override(cfg_dict: dict, dotted_key: str, raw_value: str) -> None:
    """Set a nested key from a dotted path, e.g. ``vad.hangover_ms``."""
    keys = dotted_key.split(".")
    node = cfg_dict
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = _coerce(raw_value)


def load_config(
    path: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
    set_args: list[str] | None = None,
) -> Config:
    """Load a YAML config, then apply dict and dotted-key overrides in order.

    Args:
        path: Path to the YAML file. Defaults to config/base.yaml.
        overrides: Nested overrides deep-merged on top of the file.
        set_args: ``key.subkey=value`` strings from the --set CLI flag.

    Returns:
        A Config wrapping the merged settings.

    Raises:
        ValueError: If a --set argument is not of the form ``key=value``.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        raw = _deep_merge(raw, overrides)

    for arg in set_args or []:
        if "=" not in arg:
            raise ValueError(f"--set argument must be key=value, got: '{arg}'")
        key, _, value = arg.partition("=")
        _apply_dotted_override(raw, key.strip(), value.strip())

    return Config(raw)


# Module-level default, replaced by load_config() calls from config.py.
cfg: Config = load_config()
