from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from chiplet_tuner.core.io import load_metrics, read_json
from chiplet_tuner.core.search_space import DEFAULT_HARDWARE_SEARCH_SPACE
from chiplet_tuner.core.schemas import EvaluationResult


class SimulatorAdapter(ABC):
    """Interface implemented by simulator-specific adapters."""

    @abstractmethod
    def evaluate(self, hardware: Dict[str, Any], iteration: int) -> EvaluationResult:
        """Run one hardware design point and return normalized artifacts."""

    @abstractmethod
    def schema(self) -> Dict[str, Any]:
        """Return hardware search constraints and required simulator fields."""


class GenericFileEvaluationAdapter(SimulatorAdapter):
    """Read precomputed evaluation files using the normalized detail schema."""

    def __init__(
        self,
        run_dir: Path,
        hardware_path: Path,
        metrics_path: Optional[Path] = None,
        latency_detail_path: Optional[Path] = None,
        energy_detail_path: Optional[Path] = None,
        mc_detail_path: Optional[Path] = None,
        search_space: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.hardware_path = hardware_path.resolve()
        self.metrics_path = (metrics_path or self.run_dir / "exec_res.csv").resolve()
        self.latency_detail_path = (latency_detail_path or self.run_dir / "exec_latency_detail.json").resolve()
        self.energy_detail_path = (energy_detail_path or self.run_dir / "exec_energy_detail.json").resolve()
        self.mc_detail_path = (mc_detail_path or self.run_dir / "exec_mc_detail.json").resolve()
        self.search_space = search_space or DEFAULT_HARDWARE_SEARCH_SPACE

    def evaluate(self, hardware: Dict[str, Any], iteration: int = 0) -> EvaluationResult:
        if not self.metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {self.metrics_path}")
        if not self.latency_detail_path.exists():
            raise FileNotFoundError(f"Latency detail file not found: {self.latency_detail_path}")
        if not self.energy_detail_path.exists():
            raise FileNotFoundError(f"Energy detail file not found: {self.energy_detail_path}")
        detail_files = {
            "latency": self.latency_detail_path,
            "energy": self.energy_detail_path,
        }
        if self.mc_detail_path.exists():
            detail_files["monetary_cost"] = self.mc_detail_path
        return EvaluationResult(
            run_dir=self.run_dir,
            hardware=hardware,
            metrics=load_metrics(self.metrics_path),
            detail_files=detail_files,
            raw_files={"metrics": self.metrics_path, "hardware": self.hardware_path},
        )

    def load_hardware(self) -> Dict[str, Any]:
        return read_json(self.hardware_path)

    def schema(self) -> Dict[str, Any]:
        return {
            "adapter": "generic-files",
            "design_space_kind": "bo_hierarchical",
            "search_space": self.search_space,
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
            "detail_schema": {
                "latency": "JSON keyed by resource/core, values include layerID, layerName, latencyBegin, latencyEnd, calcTime, nocTime, dramTime",
                "energy": "JSON keyed by resource/core, values include layerID, layerName, energy, calcEnergy, ubufEnergy, nocEnergy, dramEnergy",
                "monetary_cost": "Optional JSON with simulator monetary-cost breakdown, usually emitted as exec_mc_detail.json",
            },
        }
