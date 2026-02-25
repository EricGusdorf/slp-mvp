from __future__ import annotations

import json
import os
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class CacheEntry:
    value: Any
    fetched_at: float


class DiskCache:
    """
    Minimal file-based JSON cache with TTL support.

    Keys are hashed to filenames to avoid path issues.
    """

    def __init__(self, base_dir: str | Path = ".cache", default_ttl_seconds: int = 24 * 3600):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl_seconds = int(default_ttl_seconds)

    def _path_for_key(self, key: str) -> Path:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.base_dir / f"{h}.json"

    def get(self, key: str, ttl_seconds: Optional[int] = None) -> Optional[Any]:
        ttl = self.default_ttl_seconds if ttl_seconds is None else int(ttl_seconds)
        path = self._path_for_key(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            fetched_at = float(payload.get("_fetched_at", 0))
            if ttl >= 0 and (time.time() - fetched_at) > ttl:
                return None
            return payload.get("data")
        except Exception:
            # Corrupt cache; ignore.
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._path_for_key(key)
        payload = {"_fetched_at": time.time(), "data": value}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def clear(self) -> None:
        for p in self.base_dir.glob("*.json"):
            try:
                p.unlink()
            except Exception:
                pass
