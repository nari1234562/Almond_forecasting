"""
speculation_loader.py - Utility to load and validate market speculation scenarios.
"""
from pathlib import Path
from typing import Dict, List
import json

HERE = Path(__file__).resolve().parent

def get_speculation_summary(json_path: Path | str | None = None) -> List[Dict]:
    path = Path(json_path) if json_path else HERE / "external_market_speculations.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("market_factors", [])
