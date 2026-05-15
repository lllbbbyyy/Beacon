from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence


def read_json(path: Path | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path | str, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path | str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_csv_first_row(path: Path | str) -> Dict[str, float]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first = next(reader)
    result: Dict[str, float] = {}
    for key, value in first.items():
        if value in (None, ""):
            continue
        try:
            result[key] = float(value)
        except ValueError:
            pass
    return result


def load_metrics(path: Path | str) -> Dict[str, float]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        raw = read_json(path)
        return {key: float(value) for key, value in raw.items() if isinstance(value, (int, float))}
    return load_csv_first_row(path)
