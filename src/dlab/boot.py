from __future__ import annotations
import os, sys, logging
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[2]    # repo root
PKG_ROOT = Path(__file__).resolve().parent     # dlab/

_CFG: Dict[str, Any] | None = None

def load_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise SystemExit("Missing PyYAML. Run scripts/setup.ps1 or pip install -r requirements.txt")
    with open(path, "r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {})

def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "dlab.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(logfile, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger().info("Logging to %s", logfile)

def add_dll_dirs(cfg: Dict[str, Any]) -> None:
    if os.name != "nt":
        return
    for key in ("drivers_andor", "drivers_avaspec", "drivers_slm"):
        rel = cfg.get("paths", {}).get(key)
        if not rel:
            continue
        p = (ROOT / rel).resolve()
        if p.exists():
            os.add_dll_directory(str(p))

def bootstrap(config_path: Path) -> Dict[str, Any]:
    global _CFG
    cfg = load_config(config_path)
    logs_dir = Path(cfg.get("paths", {}).get("logs_dir", "logs"))
    setup_logging(ROOT / logs_dir, cfg.get("runtime", {}).get("log_level", "INFO"))
    add_dll_dirs(cfg)
    _CFG = cfg
    return cfg

def get_config() -> Dict[str, Any]:
    if _CFG is None:
        default = ROOT / "config" / "config.yaml"
        return load_config(default) if default.exists() else {}
    return _CFG
