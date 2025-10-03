"""Feature flag and dependency capability detection for Astro-AI.

Provides a single place to query which heavy / optional scientific dependencies
are available so the UI and logs can present a clean status in production.
"""
from __future__ import annotations
import importlib
from dataclasses import dataclass
from typing import Dict

OPTIONAL_LIBS = {
    "py21cmfast": ["py21cmfast"],
    "tools21cm": ["tools21cm"],
    "bagpipes": ["bagpipes"],
    "jwst_pipeline": ["jwst"],
    "astropy": ["astropy"],
}

@dataclass
class CapabilityStatus:
    name: str
    available: bool
    detail: str


def _check_mod(path: str) -> bool:
    try:
        importlib.import_module(path)
        return True
    except Exception:
        return False


def detect_capabilities() -> Dict[str, CapabilityStatus]:
    status: Dict[str, CapabilityStatus] = {}
    for logical_name, modules in OPTIONAL_LIBS.items():
        ok = any(_check_mod(m) for m in modules)
        detail = "present" if ok else "missing"
        status[logical_name] = CapabilityStatus(logical_name, ok, detail)
    return status


def summarize_status(markdown: bool = True) -> str:
    caps = detect_capabilities()
    if markdown:
        lines = ["### Capability Status", "| Feature | Available |", "|---------|-----------|"]
        for k, v in caps.items():
            lines.append(f"| {k} | {'✅' if v.available else '❌'} |")
        return "\n".join(lines)
    return ", ".join(f"{k}={v.available}" for k, v in caps.items())


def all_required_or_raise(required: list[str]) -> None:
    caps = detect_capabilities()
    missing = [r for r in required if not caps.get(r, CapabilityStatus(r, False, '')).available]
    if missing:
        raise RuntimeError(f"Missing required capabilities for strict mode: {', '.join(missing)}")
