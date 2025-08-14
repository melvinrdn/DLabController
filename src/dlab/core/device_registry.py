# src/dlab/core/device_registry.py
from __future__ import annotations
from threading import RLock
from typing import Any, Iterable

class DeviceRegistry:
    _inst: "DeviceRegistry | None" = None

    def __init__(self) -> None:
        self._lock = RLock()
        self._d: dict[str, Any] = {}

    @classmethod
    def instance(cls) -> "DeviceRegistry":
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def register(self, key: str, obj: Any) -> None:
        with self._lock:
            self._d[key] = obj

    def unregister(self, key: str) -> None:
        with self._lock:
            self._d.pop(key, None)

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._d.get(key)

    def keys(self, prefix: str | None = None) -> list[str]:
        with self._lock:
            if prefix is None:
                return list(self._d.keys())
            return [k for k in self._d.keys() if k.startswith(prefix)]

    def items(self, prefix: str | None = None) -> list[tuple[str, Any]]:
        with self._lock:
            it = list(self._d.items())
            if prefix is None:
                return it
            return [(k, v) for (k, v) in it if k.startswith(prefix)]


REGISTRY = DeviceRegistry.instance()
