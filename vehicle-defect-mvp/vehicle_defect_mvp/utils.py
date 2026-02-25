from __future__ import annotations

import re
from typing import Optional


STATE_RE = re.compile(r",\s*([A-Z]{2})\s*$")


def extract_state_abbr(consumer_location: Optional[str]) -> Optional[str]:
    """
    Extract state abbreviation from strings like "LAS VEGAS, NV".
    Returns None if not parseable or "Unknown".
    """
    if not consumer_location:
        return None
    s = consumer_location.strip()
    if not s or s.lower() == "unknown":
        return None
    m = STATE_RE.search(s.upper())
    if not m:
        return None
    return m.group(1)


def split_components(components: Optional[str]) -> list[str]:
    """
    Split components string like "ENGINE,POWER TRAIN" or "SERVICE BRAKES, AIR"
    into a normalized list.
    """
    if not components:
        return []
    parts = re.split(r"[,\|/]+", components)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.append(p.upper())
    return out


def safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def normalize_make(s: str) -> str:
    return (s or "").strip().upper()


def normalize_model(s: str) -> str:
    return (s or "").strip().upper()
