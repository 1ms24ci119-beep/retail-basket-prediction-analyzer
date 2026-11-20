from __future__ import annotations

import os
from pathlib import Path


def _load_local_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_local_env()

DATA_DIR = Path(os.environ.get("DATA_DIR", "data/generated"))
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
SEED = int(os.environ.get("SEED", "1729"))
TRANSACTIONS = int(os.environ.get("TRANSACTIONS", "1200"))
EPOCHS = int(os.environ.get("EPOCHS", "6"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
DEVICE = os.environ.get("DEVICE", "auto")
