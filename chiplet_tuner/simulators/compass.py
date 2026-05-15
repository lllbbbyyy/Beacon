from __future__ import annotations

import copy
import subprocess
from pathlib import Path
from typing import Any, Dict

from chiplet_tuner.core.io import load_csv_first_row, read_json, write_json
from chiplet_tuner.core.search_space import DEFAULT_HARDWARE_SEARCH_SPACE
from chiplet_tuner.core.schemas import EvaluationResult
from chiplet_tuner.simulators.base import SimulatorAdapter


def resolve_config_paths(config: Dict[str, Any], source_path: Path, target_dir: Path) -> Dict[str, Any]:
    """Resolve Compass config paths after copying into a run directory."""

    resolved = copy.deepcopy(config)
    source_dir = source_path.resolve().parent
    for key in ["req_generator_input_length_path", "req_generator_output_length_path"]:
        value = resolved.get(key)
        if isinstance(value, str) and value:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = (source_dir / candidate).resolve()
            resolved[key] = str(candidate)

    run_local_paths = {
        "best_mapping_save_path": "best_mapping.json",
        "search_process_save_path": "search_process.csv",
        "exec_load_path": "best_mapping.json",
        "detail_latency_save_path": "exec_latency_detail.json",
        "detail_energy_save_path": "exec_energy_detail.json",
        "detail_mc_save_path": "exec_mc_detail.json",
    }
    for key, filename in run_local_paths.items():
        if key not in resolved:
            continue
        if key.startswith("detail_") and not resolved.get(key):
            continue
        if key == "search_process_save_path" and not resolved.get(key):
            continue
        resolved[key] = str((target_dir / filename).resolve())
    return resolved


class CompassSimulatorAdapter(SimulatorAdapter):
    """Compass-specific adapter that returns simulator-independent artifacts."""

    def __init__(
        self,
        compass_root: Path,
        search_config_path: Path,
        exec_config_path: Path,
        output_root: Path,
        search_space: Dict[str, Any] | None = None,
    ) -> None:
        self.compass_root = compass_root.resolve()
        self.binary = self.compass_root / "build" / "compass"
        self.search_config_path = search_config_path.resolve()
        self.exec_config_path = exec_config_path.resolve()
        self.output_root = output_root.resolve()
        self.search_space = search_space or DEFAULT_HARDWARE_SEARCH_SPACE

    def evaluate(self, hardware: Dict[str, Any], iteration: int) -> EvaluationResult:
        if not self.binary.exists():
            raise FileNotFoundError(f"Compass binary not found: {self.binary}")

        run_dir = self.output_root / f"iter_{iteration:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        hardware_path = run_dir / "hardware.json"
        search_config_path = run_dir / "search_config.json"
        exec_config_path = run_dir / "exec_config.json"
        search_csv = run_dir / "search_res.csv"
        exec_csv = run_dir / "exec_res.csv"
        search_log = run_dir / "search.log"
        exec_log = run_dir / "exec.log"

        write_json(hardware_path, hardware)

        search_config = resolve_config_paths(read_json(self.search_config_path), self.search_config_path, run_dir)
        search_config["run_mode"] = search_config.get("run_mode", "GA")
        search_config["best_mapping_save_path"] = str((run_dir / "best_mapping.json").resolve())
        search_config.setdefault("detail_latency_save_path", "")
        search_config.setdefault("detail_energy_save_path", "")
        search_config.setdefault("detail_mc_save_path", "")
        write_json(search_config_path, search_config)

        exec_config = resolve_config_paths(read_json(self.exec_config_path), self.exec_config_path, run_dir)
        exec_config["run_mode"] = "exec"
        exec_config["exec_load_path"] = str((run_dir / "best_mapping.json").resolve())
        exec_config["detail_latency_save_path"] = str((run_dir / "exec_latency_detail.json").resolve())
        exec_config["detail_energy_save_path"] = str((run_dir / "exec_energy_detail.json").resolve())
        exec_config["detail_mc_save_path"] = str((run_dir / "exec_mc_detail.json").resolve())
        write_json(exec_config_path, exec_config)

        with search_log.open("w", encoding="utf-8") as f:
            subprocess.run(
                [str(self.binary), str(search_config_path), str(hardware_path), str(search_csv)],
                cwd=run_dir,
                stdout=f,
                stderr=f,
                check=True,
            )
        with exec_log.open("w", encoding="utf-8") as f:
            subprocess.run(
                [str(self.binary), str(exec_config_path), str(hardware_path), str(exec_csv)],
                cwd=run_dir,
                stdout=f,
                stderr=f,
                check=True,
            )

        return EvaluationResult(
            run_dir=run_dir,
            hardware=hardware,
            metrics=load_csv_first_row(exec_csv),
            detail_files={
                "latency": run_dir / "exec_latency_detail.json",
                "energy": run_dir / "exec_energy_detail.json",
                "monetary_cost": run_dir / "exec_mc_detail.json",
            },
            raw_files={
                "hardware": hardware_path,
                "search_config": search_config_path,
                "exec_config": exec_config_path,
                "search_csv": search_csv,
                "exec_csv": exec_csv,
                "search_log": search_log,
                "exec_log": exec_log,
            },
        )

    def schema(self) -> Dict[str, Any]:
        return {
            "adapter": "compass",
            "design_space_kind": "bo_hierarchical",
            "required_hardware_fields": [
                "num_chiplets",
                "chip_x",
                "chip_y",
                "nop_bw",
                "dram_bw",
                "micro_batch",
                "tensor_parall",
                "chiplets",
            ],
            "chiplet_fields": ["type", "buffer_size", "compute_units", "macs"],
            "tunable_hardware_fields": {
                "compute_spec_and_chiplet_count": ["chip_size"],
                "per_chiplet_choice": ["chiplet_type"],
                "system_params": ["dram_bw", "nop_bw", "micro_batch", "tensor_parall"],
            },
            "derived_hardware_fields": [
                "num_chiplets",
                "chip_x",
                "chip_y",
                "chiplets.compute_units",
                "chiplets.buffer_size",
                "chiplets.macs",
            ],
            "search_space": self.search_space,
            "detail_schema": {
                "latency": "exec_latency_detail.json keyed by core/resource with layer timing and communication fields",
                "energy": "exec_energy_detail.json keyed by core/resource with compute/buffer/noc/dram energy fields",
                "monetary_cost": "exec_mc_detail.json with Compass monetary-cost breakdown",
            },
        }
