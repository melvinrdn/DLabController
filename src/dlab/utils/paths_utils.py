from pathlib import Path
from dlab.boot import ROOT, get_config


def ressources_dir() -> Path:
    """Get ressources directory from config."""
    cfg = get_config() or {}
    rel = (cfg.get("paths", {}) or {}).get("ressources_dir", "ressources")
    return (ROOT / rel).resolve()

def data_dir() -> Path:
    """Get data directory from config."""
    cfg = get_config() or {}
    path = (cfg.get("paths", {}) or {}).get("data_dir", "data")
    return Path(path).resolve()