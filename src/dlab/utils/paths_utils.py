from pathlib import Path
from dlab.boot import ROOT
from dlab.utils.config_utils import cfg_get


def ressources_dir() -> Path:
    """Get ressources directory from config."""
    return (ROOT / str(cfg_get("paths.ressources_dir", "ressources"))).resolve()


def logs_dir() -> Path:
    """Get logs directory from config."""
    return (ROOT / str(cfg_get("paths.logs_dir", "logs"))).resolve()


def data_dir() -> Path:
    """Get data directory from config."""
    return (ROOT / str(cfg_get("paths.data_dir", "data"))).resolve()
