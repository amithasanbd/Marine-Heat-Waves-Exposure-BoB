"""Utility helpers for the Bay of Bengal marine heatwave reproducibility package."""
from __future__ import annotations
from pathlib import Path
import os
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("MHW_DATA_ROOT", PROJECT_ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("MHW_OUTPUT_ROOT", PROJECT_ROOT / "outputs"))

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def require_file(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required input file was not found: {path}")
    return path
