from dlab.boot import get_config


def cfg_get(path: str, default=None):
    """Get nested config value by dot-separated path."""
    cfg = get_config() or {}
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
