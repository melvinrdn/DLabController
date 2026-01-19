from __future__ import annotations
from pathlib import Path
import yaml


def read_yaml(path: Path) -> dict:
    """Read a YAML file, returning empty dict if missing or invalid."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_yaml(path: Path, data: dict) -> None:
    """Write a dict to a YAML file, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=True, allow_unicode=True)