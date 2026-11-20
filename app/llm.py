"""Compatibility for the starter's settings test; project models run locally."""
from __future__ import annotations

import os
from pathlib import Path


def _load_env() -> None:
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def settings() -> tuple[str, str, str]:
    _load_env()
    return (
        os.environ.get("LLM_BASE_URL") or "http://localhost:11434/v1",
        os.environ.get("LLM_API_KEY", ""),
        os.environ.get("LLM_MODEL", ""),
    )
