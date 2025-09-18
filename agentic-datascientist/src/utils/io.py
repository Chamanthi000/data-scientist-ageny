from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def read_csv(path: str | Path):
    import pandas as pd
    return pd.read_csv(path)
