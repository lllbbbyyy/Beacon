from __future__ import annotations

import re
from typing import Any, Dict, Sequence


def clamp_to_candidates(value: int, candidates: Sequence[int], direction: int) -> int:
    ordered = sorted(set(int(v) for v in candidates))
    if value not in ordered:
        return min(ordered, key=lambda x: abs(x - value))
    idx = ordered.index(value)
    idx = max(0, min(len(ordered) - 1, idx + direction))
    return ordered[idx]


def layer_group(name: str) -> str:
    lname = name.lower()
    if "qk" in lname or "att" in lname:
        return "attention"
    if "qkv" in lname:
        return "qkv_projection"
    if "_q" in lname or "_k" in lname or "_v" in lname:
        return "qkv_projection"
    if "out_proj" in lname or "proj" in lname:
        return "output_projection"
    if "ffn" in lname or "mlp" in lname:
        return "ffn"
    if "elt" in lname:
        return "elementwise"
    return "other"


def operator_features(layer_name: str) -> Dict[str, Any]:
    head = None
    req = None
    tiling = None
    head_match = re.search(r"head(\d+)", layer_name)
    req_match = re.search(r"req(\d+)", layer_name)
    tiling_match = re.search(r"tiling[_-]?(\d+)", layer_name)
    if head_match:
        head = int(head_match.group(1))
    if req_match:
        req = int(req_match.group(1))
    if tiling_match:
        tiling = int(tiling_match.group(1))
    return {
        "group": layer_group(layer_name),
        "head": head,
        "request": req,
        "tiling": tiling,
        "is_projection": "proj" in layer_name.lower() or "qkv" in layer_name.lower(),
        "is_elementwise": "elt" in layer_name.lower(),
    }
