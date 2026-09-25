"""Load config.json and an optional local overlay."""

from __future__ import annotations

import json
from pathlib import Path


def load_config(root: Path | None = None) -> dict:
    root = root or Path(__file__).resolve().parent.parent
    config = json.loads((root / "config" / "config.json").read_text(encoding="utf-8"))
    local = root / "config" / "local.json"
    if local.is_file():
        _merge(config, json.loads(local.read_text(encoding="utf-8")))
    config["_root"] = str(root)
    return config


def _merge(base: dict, overlay: dict) -> None:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
