"""Lightweight YAML config loader with deep-merge and dotted overrides.

This keeps configuration semantics intentionally simple:

- Load one or more YAML files.
- Deep-merge them in order (later wins).
- Apply optional dotted-key overrides passed as KEY=VALUE.

We avoid bringing in a heavyweight config system at this stage.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Union

import yaml


def load_yaml_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file into a dictionary.

    The returned object is always a mutable ``dict``; missing or empty files
    return an empty dict.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a mapping at top-level: {p}")
    return dict(data)


def deep_merge_dicts(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries (overlay wins).

    - If both values are mappings, merge recursively.
    - Otherwise, the overlay replaces the base.
    """

    out: Dict[str, Any] = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], Mapping) and isinstance(v, Mapping):
            out[k] = deep_merge_dicts(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def _parse_override_value(raw: str) -> Any:
    """Parse an override value.

    We try Python literal parsing first. Additionally handle common YAML-like
    scalars: true/false/null/none.
    """

    s = raw.strip()
    lowered = s.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None

    try:
        return ast.literal_eval(s)
    except Exception:
        return raw


def set_by_dotted_key(cfg: MutableMapping[str, Any], key: str, value: Any) -> None:
    """Set ``cfg[a][b][c] = value`` for dotted key ``a.b.c``."""

    parts = [p for p in key.split(".") if p]
    if not parts:
        raise ValueError("Empty dotted key")

    cur: MutableMapping[str, Any] = cfg
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def merge_config_files(paths: Iterable[Union[str, Path]], overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Load and merge a list of YAML config files.

    Args:
        paths: Config files in merge order. Later files override earlier ones.
        overrides: Optional iterable of ``key=value`` overrides using dotted keys.

    Returns:
        Merged config dict.
    """

    cfg: Dict[str, Any] = {}
    for p in paths:
        cfg = deep_merge_dicts(cfg, load_yaml_file(p))

    if overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override must be KEY=VALUE: {item}")
            k, v = item.split("=", 1)
            set_by_dotted_key(cfg, k.strip(), _parse_override_value(v))

    return cfg
