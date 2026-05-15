from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from chiplet_tuner.core.io import write_json
from chiplet_tuner.core.search_space import (
    all_system_param_candidates,
    chip_size_candidates,
    materialize_hardware,
    make_hardware_search_space,
)


# Baseline-aligned Compass configuration source.
# Keep these values synchronized with Compass/exp.py.
CONFIG_TEMPLATE: Dict[str, Any] = {
    "seed": 42,
    "req_generator_input_length_path": "../../config/sharegpt_input_token_lens.json",
    "req_generator_output_length_path": "../../config/sharegpt_output_token_lens.json",
    "req_gen_mode": "fixed",
    "is_chunked_prefill": False,
    "req_number": 3,
    "req_prefill_number": 2,
    "req_decode_number": 64,
    "batch_size": 66,
    "model_info": {
        "type": "gpt3",
        "n_layer": 1,
        "d_model": 4096,
        "n_head": 32,
        "d_head": 128,
        "d_ffn": 16384,
        "d_model_tiling_size": 512,
        "d_ffn_tiling_size": 2048,
    },
    "dram_num": 4,
    "run_mode": "GA",
    "best_mapping_save_path": "./best_mapping.json",
    "search_process_save_path": "./search_process.csv",
    "GA_population_size": 120,
    "GA_generations": 100,
    "exec_load_path": "./best_mapping.json",
    "detail_latency_save_path": "",
    "detail_energy_save_path": "",
    "detail_mc_save_path": "",
}

COMPASS_DATASETS = ["sharegpt", "govreport"]

COMPASS_WORKLOAD_REQ_INFO: Dict[str, Dict[str, int]] = {
    "prefill": {
        "req_number": 30,
        "req_prefill_number": 4,
        "req_decode_number": 0,
        "batch_size": 4,
    },
    "decode": {
        "req_number": 1,
        "req_prefill_number": 0,
        "req_decode_number": 128,
        "batch_size": 128,
    },
    "mixed": {
        "req_number": 4,
        "req_prefill_number": 2,
        "req_decode_number": 64,
        "batch_size": 66,
    },
}

COMPASS_SCALE_MODEL_INFO: Dict[str, Dict[str, Any]] = {
    "64": {
        "type": "gpt3",
        "n_layer": 1,
        "d_model": 4096,
        "n_head": 32,
        "d_head": 128,
        "d_ffn": 16384,
        "d_model_tiling_size": 512,
        "d_ffn_tiling_size": 2048,
    },
    "512": {
        "type": "gpt3",
        "n_layer": 1,
        "d_model": 5120,
        "n_head": 40,
        "d_head": 128,
        "d_ffn": 20480,
        "d_model_tiling_size": 640,
        "d_ffn_tiling_size": 2560,
    },
    "2048": {
        "type": "llama3",
        "n_layer": 1,
        "d_model": 8192,
        "n_head": 64,
        "d_head": 128,
        "n_kv_head": 8,
        "d_ffn": 28672,
        "d_model_tiling_size": 1024,
        "d_ffn_tiling_size": 3584,
    },
}


def build_compass_search_config(
    compass_root: Path,
    workload: str,
    dataset: str,
    compute_scale: str | int,
) -> Dict[str, Any]:
    if workload not in COMPASS_WORKLOAD_REQ_INFO:
        raise ValueError(f"Unsupported workload {workload!r}; legal values are {sorted(COMPASS_WORKLOAD_REQ_INFO)}")
    scale_key = str(compute_scale)
    if scale_key not in COMPASS_SCALE_MODEL_INFO:
        raise ValueError(f"Unsupported compute_scale {compute_scale!r}; legal values are {sorted(COMPASS_SCALE_MODEL_INFO)}")

    input_lens_path, output_lens_path = dataset_token_length_paths(compass_root, dataset)
    config = copy.deepcopy(CONFIG_TEMPLATE)
    # exp.py uses ../../config/{dataset}_*.json from its generated experiment directory.
    # The tuner writes configs under tmp/runs, so absolute paths preserve the same dataset
    # selection without depending on a Compass-relative working directory layout.
    config["req_generator_input_length_path"] = str(input_lens_path)
    config["req_generator_output_length_path"] = str(output_lens_path)
    config.update(copy.deepcopy(COMPASS_WORKLOAD_REQ_INFO[workload]))
    config["model_info"] = copy.deepcopy(COMPASS_SCALE_MODEL_INFO[scale_key])
    return config


def build_compass_exec_config(search_config: Dict[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(search_config)
    config["seed"] = int(config["seed"]) + 1
    config["req_number"] = int(config["req_number"]) * 10
    config["run_mode"] = "exec"
    config["detail_latency_save_path"] = "./exec_latency_detail.json"
    config["detail_energy_save_path"] = "./exec_energy_detail.json"
    config["detail_mc_save_path"] = "./exec_mc_detail.json"
    return config


def write_compass_config_pair(
    compass_root: Path,
    output_dir: Path,
    workload: str,
    dataset: str,
    compute_scale: str | int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    search_config = build_compass_search_config(
        compass_root=compass_root,
        workload=workload,
        dataset=dataset,
        compute_scale=compute_scale,
    )
    exec_config = build_compass_exec_config(search_config)
    search_path = output_dir / "search_config.json"
    exec_path = output_dir / "exec_config.json"
    write_json(search_path, search_config)
    write_json(exec_path, exec_config)
    return search_path, exec_path


def build_compass_initial_hardware(
    task_type: str,
    compute_scale: str | int,
    accelerator_compute_budget: Optional[int] = None,
    chip_size: Optional[int] = None,
    chiplet_type: str = "ws",
) -> Dict[str, Any]:
    search_space = make_hardware_search_space(
        task_type=task_type,
        compute_scale=compute_scale,
        accelerator_compute_budget=accelerator_compute_budget,
    )
    system_params = {
        key: _middle_candidate(candidates)
        for key, candidates in all_system_param_candidates(search_space).items()
        if candidates
    }
    legal_chip_sizes = chip_size_candidates(search_space, None)
    if not legal_chip_sizes:
        raise ValueError("Cannot generate initial hardware because no legal chip_size candidates exist.")
    selected_chip_size = int(chip_size) if chip_size is not None else int(_middle_candidate(legal_chip_sizes))
    template = _synthetic_hardware_template(system_params)
    return materialize_hardware(
        template=template,
        chip_size=selected_chip_size,
        system_params=system_params,
        search_space=search_space,
        chiplet_type_strategy="uniform",
        chiplet_type_fill=chiplet_type,
    )


def write_compass_initial_hardware(
    output_dir: Path,
    task_type: str,
    compute_scale: str | int,
    accelerator_compute_budget: Optional[int] = None,
    chip_size: Optional[int] = None,
    chiplet_type: str = "ws",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    hardware = build_compass_initial_hardware(
        task_type=task_type,
        compute_scale=compute_scale,
        accelerator_compute_budget=accelerator_compute_budget,
        chip_size=chip_size,
        chiplet_type=chiplet_type,
    )
    hardware_path = output_dir / "hardware_initial.json"
    write_json(hardware_path, hardware)
    return hardware_path


def _synthetic_hardware_template(system_params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "num_chiplets": 1,
        "chip_x": 1,
        "chip_y": 1,
        "chiplets": [],
        **system_params,
    }


def _middle_candidate(candidates: Sequence[Any]) -> Any:
    if not candidates:
        raise ValueError("Cannot choose a middle candidate from an empty candidate list.")
    return list(candidates)[len(candidates) // 2]


def dataset_token_length_paths(compass_root: Path, dataset: str) -> tuple[Path, Path]:
    if dataset not in COMPASS_DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}; legal values are {sorted(COMPASS_DATASETS)}")
    config_dir = compass_root / "config"
    input_lens_path = (config_dir / f"{dataset}_input_token_lens.json").resolve()
    output_lens_path = (config_dir / f"{dataset}_output_token_lens.json").resolve()
    missing = [str(path) for path in [input_lens_path, output_lens_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Dataset {dataset!r} is missing token length files: {', '.join(missing)}"
        )
    return input_lens_path, output_lens_path
