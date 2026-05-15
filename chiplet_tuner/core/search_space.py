from __future__ import annotations

import copy
import math
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


CHIPLET_COMPUTE_SPECS: List[Dict[str, Any]] = [
    {
        "chip_size": 0,
        "name": "small",
        "compute_units": 1024,
        "buffer_size": 2048,
        "macs_by_type": {"ws": "[4,4,8,8]", "os": "[4,4,8,8]"},
    },
    {
        "chip_size": 1,
        "name": "medium",
        "compute_units": 4096,
        "buffer_size": 8192,
        "macs_by_type": {"ws": "[8,8,8,8]", "os": "[8,8,8,8]"},
    },
    {
        "chip_size": 2,
        "name": "large",
        "compute_units": 16384,
        "buffer_size": 32768,
        "macs_by_type": {"ws": "[8,8,16,16]", "os": "[8,8,16,16]"},
    },
]

CHIPLET_TYPE_CANDIDATES = ["ws", "os"]

TASK_MICRO_BATCH_OPTIONS = {
    "prefill": [1, 2, 4],
    "mixed": [1, 2, 3, 6, 11, 22, 33, 66],
    "decode": [1, 2, 4, 8, 16, 32, 64, 128],
    "serving_prefill": [1],
    "serving": [1, 2, 4, 8, 16, 32, 64, 128],
}

SYSTEM_PARAM_SEARCH_SPACE = {
    "dram_bw": [16, 32, 64, 128, 256],
    "nop_bw": [32, 64, 128, 256, 512],
    "micro_batch": TASK_MICRO_BATCH_OPTIONS["decode"],
    "tensor_parall": [4, 8, 16, 32, 64],
}

BO_SHAPE_LIST_BY_SCALE = {
    "64": [(8, 4), (4, 2), (2, 1)],
    "512": [(16, 16), (8, 8), (4, 4)],
    "2048": [(32, 32), (16, 16), (8, 8)],
}

DEFAULT_HARDWARE_SEARCH_SPACE: Dict[str, Any] = {
    "design_space_kind": "bo_hierarchical",
    "accelerator_compute_budget": None,
    "compute_scale": None,
    "chip_specs": CHIPLET_COMPUTE_SPECS,
    "chip_size_candidates": [spec["chip_size"] for spec in CHIPLET_COMPUTE_SPECS],
    "chiplet_type_candidates": CHIPLET_TYPE_CANDIDATES,
    "system_params": SYSTEM_PARAM_SEARCH_SPACE,
    "bo_shape_list_by_scale": {scale: [list(shape) for shape in shapes] for scale, shapes in BO_SHAPE_LIST_BY_SCALE.items()},
    "tunable_categories": {
        "compute_spec_and_chiplet_count": {
            "parameter": "chip_size",
            "description": (
                "Select one per-chip compute spec. The accelerator compute budget determines "
                "num_chiplets/chip_x/chip_y; compute_units, buffer_size, and macs are derived."
            ),
        },
        "per_chiplet_choice": {
            "parameter": "chiplet_type",
            "candidates": CHIPLET_TYPE_CANDIDATES,
            "description": "After chiplet count is fixed, choose each chiplet type independently.",
        },
        "system_params": {
            "parameters": ["dram_bw", "nop_bw", "micro_batch", "tensor_parall"],
        },
    },
    # Backward-compatible aliases for callers that only need system parameters.
    "dram_bw": SYSTEM_PARAM_SEARCH_SPACE["dram_bw"],
    "nop_bw": SYSTEM_PARAM_SEARCH_SPACE["nop_bw"],
    "micro_batch": SYSTEM_PARAM_SEARCH_SPACE["micro_batch"],
    "tensor_parall": SYSTEM_PARAM_SEARCH_SPACE["tensor_parall"],
}


def make_hardware_search_space(
    task_type: str = "decode",
    compute_scale: Optional[int | str] = None,
    accelerator_compute_budget: Optional[int] = None,
) -> Dict[str, Any]:
    search_space = copy.deepcopy(DEFAULT_HARDWARE_SEARCH_SPACE)
    micro_batch = micro_batch_options_for_task(task_type)
    search_space["system_params"]["micro_batch"] = micro_batch
    search_space["micro_batch"] = micro_batch
    search_space["task_type"] = task_type
    if compute_scale is not None:
        scale_key = str(compute_scale)
        if scale_key not in search_space["bo_shape_list_by_scale"]:
            raise ValueError(
                f"Unsupported compute_scale={compute_scale}; "
                f"legal values are {sorted(search_space['bo_shape_list_by_scale'])}"
            )
        search_space["compute_scale"] = scale_key
    if accelerator_compute_budget is not None:
        search_space["accelerator_compute_budget"] = int(accelerator_compute_budget)
    return search_space


def micro_batch_options_for_task(task_type: str) -> List[int]:
    if task_type in TASK_MICRO_BATCH_OPTIONS:
        return list(TASK_MICRO_BATCH_OPTIONS[task_type])
    elif str(task_type).startswith("serving"):
        return list(TASK_MICRO_BATCH_OPTIONS["serving"])
    raise ValueError(
        f"Unsupported task_type={task_type!r}; "
        f"legal values are {sorted(TASK_MICRO_BATCH_OPTIONS)} or serving*"
    )


def system_param_candidates(search_space: Mapping[str, Any], key: str) -> List[Any]:
    system_params = search_space.get("system_params", {})
    if isinstance(system_params, Mapping) and isinstance(system_params.get(key), Sequence):
        return list(system_params[key])
    candidates = search_space.get(key)
    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
        return list(candidates)
    return []


def all_system_param_candidates(search_space: Mapping[str, Any]) -> Dict[str, List[Any]]:
    return {
        key: system_param_candidates(search_space, key)
        for key in ["dram_bw", "nop_bw", "micro_batch", "tensor_parall"]
    }


def chip_type_candidates(search_space: Mapping[str, Any]) -> List[str]:
    candidates = search_space.get("chiplet_type_candidates", search_space.get("type", CHIPLET_TYPE_CANDIDATES))
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return list(CHIPLET_TYPE_CANDIDATES)
    values = [str(item) for item in candidates if str(item)]
    return values or list(CHIPLET_TYPE_CANDIDATES)


def chip_specs(search_space: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw_specs = search_space.get("chip_specs")
    if isinstance(raw_specs, Sequence) and not isinstance(raw_specs, (str, bytes)):
        specs = [copy.deepcopy(spec) for spec in raw_specs if isinstance(spec, Mapping)]
        if specs:
            return specs
    compute_candidates = search_space.get("compute_units", [1024, 4096, 16384])
    buffer_candidates = search_space.get("buffer_size", [2048, 8192, 32768])
    if not isinstance(compute_candidates, Sequence) or isinstance(compute_candidates, (str, bytes)):
        compute_candidates = [1024, 4096, 16384]
    if not isinstance(buffer_candidates, Sequence) or isinstance(buffer_candidates, (str, bytes)):
        buffer_candidates = [2048, 8192, 32768]
    specs: List[Dict[str, Any]] = []
    for idx, compute_units in enumerate(compute_candidates):
        buffer_size = buffer_candidates[min(idx, len(buffer_candidates) - 1)]
        specs.append(
            {
                "chip_size": idx,
                "name": f"legacy_{idx}",
                "compute_units": int(compute_units),
                "buffer_size": int(buffer_size),
                "macs_by_type": {},
            }
        )
    return specs


def chip_spec_by_size(search_space: Mapping[str, Any], chip_size: int) -> Dict[str, Any]:
    for spec in chip_specs(search_space):
        if int(spec.get("chip_size", -1)) == int(chip_size):
            return spec
    raise ValueError(f"Unknown chip_size {chip_size}; legal sizes are {chip_size_candidates(search_space, None)}")


def infer_total_compute_budget(hardware: Mapping[str, Any]) -> Optional[int]:
    chiplets = hardware.get("chiplets", [])
    if not isinstance(chiplets, Sequence) or isinstance(chiplets, (str, bytes)):
        return None
    total = 0
    for chip in chiplets:
        if not isinstance(chip, Mapping):
            continue
        compute_units = chip.get("compute_units")
        if compute_units is None:
            continue
        total += int(float(compute_units))
    return total or None


def resolve_accelerator_compute_budget(
    search_space: Mapping[str, Any],
    hardware: Optional[Mapping[str, Any]] = None,
) -> Optional[int]:
    explicit = search_space.get("accelerator_compute_budget")
    if explicit not in (None, ""):
        return int(float(explicit))

    scale = search_space.get("compute_scale")
    if scale not in (None, ""):
        shapes = _preset_shapes(search_space, str(scale))
        specs = chip_specs(search_space)
        if shapes and specs:
            h, w = shapes[0]
            return int(h) * int(w) * int(specs[0]["compute_units"])

    if hardware is not None:
        return infer_total_compute_budget(hardware)
    return None


def infer_chip_size(hardware: Mapping[str, Any], search_space: Mapping[str, Any]) -> Optional[int]:
    chiplets = hardware.get("chiplets", [])
    if not isinstance(chiplets, Sequence) or isinstance(chiplets, (str, bytes)) or not chiplets:
        return None
    compute_counts: Counter[int] = Counter()
    buffer_counts: Counter[int] = Counter()
    for chip in chiplets:
        if not isinstance(chip, Mapping):
            continue
        if chip.get("compute_units") is not None:
            compute_counts[int(float(chip["compute_units"]))] += 1
        if chip.get("buffer_size") is not None:
            buffer_counts[int(float(chip["buffer_size"]))] += 1
    common_compute = compute_counts.most_common(1)[0][0] if compute_counts else None
    common_buffer = buffer_counts.most_common(1)[0][0] if buffer_counts else None
    for spec in chip_specs(search_space):
        if common_compute == int(spec.get("compute_units", -1)) and (
            common_buffer is None or common_buffer == int(spec.get("buffer_size", -1))
        ):
            return int(spec["chip_size"])
    return None


def chip_size_candidates(
    search_space: Mapping[str, Any],
    hardware: Optional[Mapping[str, Any]],
) -> List[int]:
    return [int(item["chip_size"]) for item in shape_candidates(search_space, hardware)]


def shape_candidates(
    search_space: Mapping[str, Any],
    hardware: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    specs = chip_specs(search_space)
    scale = search_space.get("compute_scale")
    if scale not in (None, ""):
        shapes = _preset_shapes(search_space, str(scale))
        if shapes:
            return _candidates_from_shapes(specs, shapes)

    raw_shapes = search_space.get("shape_list", search_space.get("SHAPE_LIST"))
    if isinstance(raw_shapes, Sequence) and raw_shapes and not isinstance(raw_shapes, (str, bytes)):
        shapes = [_parse_shape_item(item) for item in raw_shapes]
        return _candidates_from_shapes(specs, shapes)

    budget = resolve_accelerator_compute_budget(search_space, hardware)
    candidates: List[Dict[str, Any]] = []
    if budget:
        for spec in specs:
            per_chip_compute = int(spec["compute_units"])
            if per_chip_compute <= 0 or budget % per_chip_compute != 0:
                continue
            num_chiplets = budget // per_chip_compute
            h, w = factor_grid(num_chiplets)
            candidates.append(
                {
                    "chip_size": int(spec["chip_size"]),
                    "chip_spec": copy.deepcopy(spec),
                    "shape": [h, w],
                    "chip_y": h,
                    "chip_x": w,
                    "num_chiplets": num_chiplets,
                    "accelerator_compute_budget": budget,
                    "budget_exact": True,
                }
            )
    if candidates:
        return candidates

    current_size = infer_chip_size(hardware, search_space) if hardware is not None else None
    if current_size is not None and hardware is not None:
        h = int(hardware.get("chip_y", 1))
        w = int(hardware.get("chip_x", hardware.get("num_chiplets", 1)))
        num_chiplets = int(hardware.get("num_chiplets", h * w))
        spec = chip_spec_by_size(search_space, current_size)
        return [
            {
                "chip_size": current_size,
                "chip_spec": copy.deepcopy(spec),
                "shape": [h, w],
                "chip_y": h,
                "chip_x": w,
                "num_chiplets": num_chiplets,
                "accelerator_compute_budget": num_chiplets * int(spec["compute_units"]),
                "budget_exact": True,
            }
        ]
    return []


def factor_grid(num_chiplets: int) -> Tuple[int, int]:
    if num_chiplets <= 0:
        raise ValueError(f"num_chiplets must be positive, got {num_chiplets}")
    root = int(math.sqrt(num_chiplets))
    for width in range(root, 0, -1):
        if num_chiplets % width == 0:
            return num_chiplets // width, width
    return num_chiplets, 1


def step_chip_size(
    hardware: Mapping[str, Any],
    search_space: Mapping[str, Any],
    direction: int,
) -> int:
    current = infer_chip_size(hardware, search_space)
    candidates = chip_size_candidates(search_space, hardware)
    if not candidates:
        raise ValueError("No legal chip_size candidates can be derived from the design space.")
    ordered = sorted(set(candidates))
    if current not in ordered:
        nearest = min(ordered, key=lambda item: abs(item - int(current or ordered[0])))
        idx = ordered.index(nearest)
    else:
        idx = ordered.index(current)
    idx = max(0, min(len(ordered) - 1, idx + direction))
    return ordered[idx]


def materialize_hardware(
    template: Mapping[str, Any],
    chip_size: int,
    chiplet_types: Optional[Sequence[Any]] = None,
    system_params: Optional[Mapping[str, Any]] = None,
    search_space: Optional[Mapping[str, Any]] = None,
    chiplet_type_strategy: str = "preserve_prefix",
    chiplet_type_fill: Optional[Any] = None,
) -> Dict[str, Any]:
    search_space = search_space or DEFAULT_HARDWARE_SEARCH_SPACE
    shape = shape_candidate_for_chip_size(search_space, template, chip_size)
    if shape is None:
        raise ValueError(f"chip_size={chip_size} is not legal for the current accelerator compute budget.")

    hardware = copy.deepcopy(dict(template))
    hardware["num_chiplets"] = int(shape["num_chiplets"])
    hardware["chip_y"] = int(shape["chip_y"])
    hardware["chip_x"] = int(shape["chip_x"])

    for key, candidates in all_system_param_candidates(search_space).items():
        source_value = (system_params or {}).get(key, hardware.get(key))
        if candidates:
            hardware[key] = coerce_to_candidates(source_value, hardware.get(key), candidates)
        elif source_value is not None:
            hardware[key] = source_value

    legal_types = chip_type_candidates(search_space)
    normalized_types = _normalized_chiplet_types(
        template=template,
        chiplet_types=chiplet_types,
        count=int(shape["num_chiplets"]),
        legal_types=legal_types,
        strategy=chiplet_type_strategy,
        fill_type=chiplet_type_fill,
    )
    hardware["chiplets"] = [
        get_chiplet_spec(chip_type, chip_size, search_space) for chip_type in normalized_types
    ]
    return hardware


def normalize_hardware_to_design_space(
    proposed: Mapping[str, Any],
    current: Mapping[str, Any],
    search_space: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    search_space = search_space or DEFAULT_HARDWARE_SEARCH_SPACE
    notes: List[str] = []
    if not isinstance(proposed, Mapping):
        return copy.deepcopy(dict(current)), ["proposal is not a hardware object; kept current hardware"]

    current_chip_size = infer_chip_size(current, search_space)
    requested_chip_size = _requested_chip_size(proposed, current_chip_size, search_space, notes)
    legal_sizes = chip_size_candidates(search_space, current)
    if not legal_sizes:
        return copy.deepcopy(dict(current)), ["no legal chip_size candidates can be derived; kept current hardware"]
    if requested_chip_size is None:
        requested_chip_size = current_chip_size if current_chip_size in legal_sizes else legal_sizes[0]
    if requested_chip_size not in legal_sizes:
        nearest = min(legal_sizes, key=lambda item: abs(item - int(requested_chip_size)))
        notes.append(f"clamped chip_size={requested_chip_size} to legal value {nearest}")
        requested_chip_size = nearest

    system_params: Dict[str, Any] = {}
    for key, candidates in all_system_param_candidates(search_space).items():
        if key in proposed:
            system_params[key] = coerce_to_candidates(proposed[key], current.get(key), candidates, key=key, notes=notes)
        elif key in current:
            system_params[key] = coerce_to_candidates(current[key], current.get(key), candidates, key=key, notes=notes)

    chiplet_types = _requested_chiplet_types(proposed)
    if chiplet_types is None:
        chiplet_types = _current_chiplet_types(current)
    chiplet_type_strategy = _requested_chiplet_type_strategy(proposed)
    chiplet_type_fill = _requested_chiplet_type_fill(proposed)

    updated = materialize_hardware(
        template=current,
        chip_size=int(requested_chip_size),
        chiplet_types=chiplet_types,
        system_params=system_params,
        search_space=search_space,
        chiplet_type_strategy=chiplet_type_strategy,
        chiplet_type_fill=chiplet_type_fill,
    )
    _add_materialization_notes(current, updated, search_space, notes)
    return updated, notes


def get_chiplet_spec(chip_type: Any, chip_size: int, search_space: Mapping[str, Any]) -> Dict[str, Any]:
    legal_types = chip_type_candidates(search_space)
    chip_type_str = str(chip_type)
    if chip_type_str not in legal_types:
        chip_type_str = legal_types[0]
    spec = chip_spec_by_size(search_space, chip_size)
    macs_by_type = spec.get("macs_by_type", {})
    macs = macs_by_type.get(chip_type_str) if isinstance(macs_by_type, Mapping) else None
    if macs is None:
        raw_macs = spec.get("macs")
        if isinstance(raw_macs, Mapping):
            macs = raw_macs.get(chip_type_str)
        elif raw_macs is not None:
            macs = raw_macs
    result = {
        "type": chip_type_str,
        "buffer_size": int(spec["buffer_size"]),
        "compute_units": int(spec["compute_units"]),
    }
    if macs is not None:
        result["macs"] = macs
    return result


def shape_candidate_for_chip_size(
    search_space: Mapping[str, Any],
    hardware: Optional[Mapping[str, Any]],
    chip_size: int,
) -> Optional[Dict[str, Any]]:
    for candidate in shape_candidates(search_space, hardware):
        if int(candidate["chip_size"]) == int(chip_size):
            return copy.deepcopy(candidate)
    return None


def coerce_to_candidates(
    value: Any,
    current_value: Any,
    candidates: Sequence[Any],
    key: str = "",
    notes: Optional[List[str]] = None,
) -> Any:
    if not candidates:
        return value
    if value in candidates:
        return value
    numeric = _numeric_candidates(candidates)
    if numeric:
        try:
            coerced = min(numeric, key=lambda item: abs(item - float(value)))
            result: Any = int(coerced) if float(coerced).is_integer() else coerced
            if notes is not None:
                notes.append(f"clamped {key or 'value'}={value} to nearest legal value {result}")
            return result
        except (TypeError, ValueError):
            pass
    if current_value in candidates:
        if notes is not None:
            notes.append(f"kept current {key or 'value'} because proposed value {value} is invalid")
        return current_value
    if notes is not None:
        notes.append(f"replaced invalid {key or 'value'}={value} with first legal value {candidates[0]}")
    return copy.deepcopy(candidates[0])


def _numeric_candidates(candidates: Sequence[Any]) -> List[float]:
    numeric: List[float] = []
    for item in candidates:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return []
        numeric.append(float(item))
    return numeric


def _preset_shapes(search_space: Mapping[str, Any], scale: str) -> List[Tuple[int, int]]:
    raw_presets = search_space.get("bo_shape_list_by_scale", BO_SHAPE_LIST_BY_SCALE)
    if not isinstance(raw_presets, Mapping):
        return []
    raw_shapes = raw_presets.get(str(scale), raw_presets.get(int(scale) if str(scale).isdigit() else scale, []))
    if not isinstance(raw_shapes, Sequence) or isinstance(raw_shapes, (str, bytes)):
        return []
    return [_parse_shape_item(item) for item in raw_shapes]


def _parse_shape_item(item: Any) -> Tuple[int, int]:
    if isinstance(item, Mapping):
        return int(item["H"]), int(item["W"])
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
        return int(item[0]), int(item[1])
    raise ValueError(f"Invalid shape item: {item!r}")


def _candidates_from_shapes(specs: Sequence[Mapping[str, Any]], shapes: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for idx, shape in enumerate(shapes):
        if idx >= len(specs):
            break
        h, w = int(shape[0]), int(shape[1])
        spec = copy.deepcopy(dict(specs[idx]))
        candidates.append(
            {
                "chip_size": int(spec["chip_size"]),
                "chip_spec": spec,
                "shape": [h, w],
                "chip_y": h,
                "chip_x": w,
                "num_chiplets": h * w,
                "accelerator_compute_budget": h * w * int(spec["compute_units"]),
                "budget_exact": True,
            }
        )
    return candidates


def _requested_chip_size(
    proposed: Mapping[str, Any],
    current_chip_size: Optional[int],
    search_space: Mapping[str, Any],
    notes: List[str],
) -> Optional[int]:
    for key in ["chip_size", "chip_spec", "compute_spec"]:
        if key in proposed:
            return _parse_chip_size_value(proposed[key], search_space, notes)
    if "compute_units" in proposed or "buffer_size" in proposed:
        parsed = _parse_chip_size_from_spec_fields(proposed, search_space)
        if parsed is not None:
            notes.append("interpreted proposed top-level compute_units/buffer_size as chip_size change")
            return parsed
    chiplets = proposed.get("chiplets")
    if isinstance(chiplets, Sequence) and not isinstance(chiplets, (str, bytes)) and chiplets:
        parsed = _parse_chip_size_from_chiplets(chiplets, search_space)
        if parsed is not None:
            if parsed != current_chip_size:
                notes.append("interpreted proposed chiplet compute_units/buffer_size as chip_size change")
            return parsed
    return current_chip_size


def _parse_chip_size_value(value: Any, search_space: Mapping[str, Any], notes: List[str]) -> Optional[int]:
    if isinstance(value, Mapping):
        if "chip_size" in value:
            return int(value["chip_size"])
        return _parse_chip_size_from_spec_fields(value, search_space)
    if isinstance(value, str):
        for spec in chip_specs(search_space):
            if value == str(spec.get("name")):
                return int(spec["chip_size"])
        try:
            return int(value)
        except ValueError:
            notes.append(f"ignored unknown chip_size value {value!r}")
            return None
    if isinstance(value, (int, float)):
        return int(value)
    notes.append(f"ignored invalid chip_size value {value!r}")
    return None


def _parse_chip_size_from_spec_fields(payload: Mapping[str, Any], search_space: Mapping[str, Any]) -> Optional[int]:
    compute_units = payload.get("compute_units")
    buffer_size = payload.get("buffer_size")
    for spec in chip_specs(search_space):
        compute_matches = compute_units is None or int(float(compute_units)) == int(spec["compute_units"])
        buffer_matches = buffer_size is None or int(float(buffer_size)) == int(spec["buffer_size"])
        if compute_matches and buffer_matches:
            return int(spec["chip_size"])
    return None


def _parse_chip_size_from_chiplets(chiplets: Sequence[Any], search_space: Mapping[str, Any]) -> Optional[int]:
    compute_counts: Counter[int] = Counter()
    buffer_counts: Counter[int] = Counter()
    for chip in chiplets:
        if not isinstance(chip, Mapping):
            continue
        if chip.get("compute_units") is not None:
            compute_counts[int(float(chip["compute_units"]))] += 1
        if chip.get("buffer_size") is not None:
            buffer_counts[int(float(chip["buffer_size"]))] += 1
    if not compute_counts and not buffer_counts:
        return None
    payload: Dict[str, Any] = {}
    if compute_counts:
        payload["compute_units"] = compute_counts.most_common(1)[0][0]
    if buffer_counts:
        payload["buffer_size"] = buffer_counts.most_common(1)[0][0]
    return _parse_chip_size_from_spec_fields(payload, search_space)


def _requested_chiplet_types(proposed: Mapping[str, Any]) -> Optional[List[Any]]:
    raw_types = proposed.get("chiplet_types", proposed.get("chiplet_type"))
    if isinstance(raw_types, str):
        return [raw_types]
    if isinstance(raw_types, Sequence) and not isinstance(raw_types, (str, bytes)):
        return list(raw_types)
    chiplets = proposed.get("chiplets")
    if isinstance(chiplets, Sequence) and not isinstance(chiplets, (str, bytes)):
        types = [chip.get("type") for chip in chiplets if isinstance(chip, Mapping) and chip.get("type") is not None]
        return types or None
    return None


def _requested_chiplet_type_strategy(proposed: Mapping[str, Any]) -> str:
    raw = proposed.get("chiplet_type_strategy", proposed.get("type_strategy", "preserve_prefix"))
    return _normalize_chiplet_type_strategy(raw)


def _requested_chiplet_type_fill(proposed: Mapping[str, Any]) -> Optional[Any]:
    for key in ["chiplet_type_fill", "fill_chiplet_type", "uniform_chiplet_type"]:
        if proposed.get(key) is not None:
            return proposed[key]
    strategy = _requested_chiplet_type_strategy(proposed)
    if strategy == "uniform":
        raw_types = proposed.get("chiplet_types", proposed.get("chiplet_type"))
        if isinstance(raw_types, str):
            return raw_types
        if isinstance(raw_types, Sequence) and not isinstance(raw_types, (str, bytes)) and raw_types:
            return raw_types[0]
    return None


def _current_chiplet_types(hardware: Mapping[str, Any]) -> List[Any]:
    chiplets = hardware.get("chiplets", [])
    if not isinstance(chiplets, Sequence) or isinstance(chiplets, (str, bytes)):
        return []
    return [chip.get("type") for chip in chiplets if isinstance(chip, Mapping)]


def _normalized_chiplet_types(
    template: Mapping[str, Any],
    chiplet_types: Optional[Sequence[Any]],
    count: int,
    legal_types: Sequence[str],
    strategy: str = "preserve_prefix",
    fill_type: Optional[Any] = None,
) -> List[str]:
    source = list(chiplet_types or _current_chiplet_types(template))
    legal = [str(item) for item in legal_types]
    if not legal:
        raise ValueError("chiplet type candidate list cannot be empty.")
    strategy = _normalize_chiplet_type_strategy(strategy)
    requested_fill = str(fill_type) if fill_type is not None else None
    if requested_fill is not None and requested_fill not in legal:
        raise ValueError(f"Invalid chiplet_type_fill {requested_fill!r}; legal values are {legal}")
    default_type = requested_fill or _most_common_legal_type(source, legal) or legal[0]
    if strategy == "uniform":
        return [default_type for _ in range(count)]
    if strategy == "majority":
        majority_type = _most_common_legal_type(source, legal) or default_type
        return [majority_type for _ in range(count)]
    normalized: List[str] = []
    for idx in range(count):
        value = str(source[idx]) if idx < len(source) and source[idx] is not None else default_type
        normalized.append(value if value in legal else default_type)
    return normalized


def _normalize_chiplet_type_strategy(raw_strategy: Any) -> str:
    strategy = str(raw_strategy or "preserve_prefix").strip().lower()
    aliases = {
        "preserve": "preserve_prefix",
        "preserve_prefix": "preserve_prefix",
        "prefix": "preserve_prefix",
        "prefix_majority": "preserve_prefix",
        "current": "preserve_prefix",
        "majority": "majority",
        "majority_fill": "majority",
        "most_common": "majority",
        "mainstream": "majority",
        "uniform": "uniform",
        "all": "uniform",
        "set_all": "uniform",
        "same": "uniform",
    }
    if strategy not in aliases:
        raise ValueError(
            f"Unsupported chiplet_type_strategy {raw_strategy!r}; "
            "legal values are preserve_prefix, majority, uniform"
        )
    return aliases[strategy]


def _most_common_legal_type(values: Sequence[Any], legal_types: Sequence[str]) -> Optional[str]:
    counts: Counter[str] = Counter(str(item) for item in values if str(item) in legal_types)
    return counts.most_common(1)[0][0] if counts else None


def _add_materialization_notes(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    search_space: Mapping[str, Any],
    notes: List[str],
) -> None:
    old_size = infer_chip_size(old, search_space)
    new_size = infer_chip_size(new, search_space)
    if old_size != new_size:
        old_spec = chip_spec_by_size(search_space, old_size) if old_size is not None else {}
        new_spec = chip_spec_by_size(search_space, new_size) if new_size is not None else {}
        notes.append(
            "chip_size changed "
            f"{old_size}->{new_size}; per_chip_compute "
            f"{old_spec.get('compute_units')}->{new_spec.get('compute_units')}; "
            f"per_chip_buffer {old_spec.get('buffer_size')}->{new_spec.get('buffer_size')}"
        )
    if old.get("num_chiplets") != new.get("num_chiplets"):
        notes.append(
            f"num_chiplets changed {old.get('num_chiplets')}->{new.get('num_chiplets')} "
            "to preserve accelerator compute budget"
        )
