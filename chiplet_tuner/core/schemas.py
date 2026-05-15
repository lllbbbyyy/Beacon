from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationResult:
    """Simulator-independent description of one evaluated design point.

    A simulator adapter is responsible for producing this object.  Downstream
    agents only depend on these fields, not on a specific simulator.
    """

    run_dir: Path
    hardware: Dict[str, Any]
    metrics: Dict[str, float]
    detail_files: Dict[str, Path]
    raw_files: Dict[str, Path] = field(default_factory=dict)


@dataclass
class LayerLoad:
    layer_id: int
    layer_name: str
    group: str
    occurrences: int = 0
    cores: List[str] = field(default_factory=list)
    batches: List[int] = field(default_factory=list)
    latency_sum: float = 0.0
    latency_max: float = 0.0
    critical_end: float = 0.0
    calc_time: float = 0.0
    noc_time: float = 0.0
    dram_time: float = 0.0
    energy: float = 0.0
    calc_energy: float = 0.0
    ubuf_energy: float = 0.0
    noc_energy: float = 0.0
    dram_energy: float = 0.0
    time_breakdown: Dict[str, float] = field(default_factory=dict)
    energy_breakdown: Dict[str, float] = field(default_factory=dict)
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    dominant_dimension: str = "unknown"
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "layer_name": self.layer_name,
            "layerID": self.layer_id,
            "layerName": self.layer_name,
            "group": self.group,
            "occurrences": self.occurrences,
            "cores": sorted(set(self.cores), key=_core_sort_key),
            "batches": sorted(set(self.batches)),
            "latency_sum": self.latency_sum,
            "latency_max": self.latency_max,
            "critical_end": self.critical_end,
            "calc_time": self.calc_time,
            "noc_time": self.noc_time,
            "dram_time": self.dram_time,
            "energy": self.energy,
            "calc_energy": self.calc_energy,
            "ubuf_energy": self.ubuf_energy,
            "noc_energy": self.noc_energy,
            "dram_energy": self.dram_energy,
            "time_breakdown": self.time_breakdown,
            "energy_breakdown": self.energy_breakdown,
            "dimension_scores": self.dimension_scores,
            "dominant_dimension": self.dominant_dimension,
            "features": self.features,
        }


@dataclass
class ToolResult:
    name: str
    payload: Dict[str, Any]
    generated_files: Dict[str, str] = field(default_factory=dict)


@dataclass
class ModelAnalysisResult:
    metrics: Dict[str, float]
    candidate_layers: List[Dict[str, Any]]
    generated_files: Dict[str, str]
    summary: str
    global_findings: List[str] = field(default_factory=list)
    selected_views: List[str] = field(default_factory=list)
    analysis_base: Dict[str, Any] = field(default_factory=dict)
    llm_notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BottleneckState:
    primary_impact: str
    dominant_root_cause: str
    layer_diagnoses: List[Dict[str, Any]]
    retrieval_description: str
    root_cause_summary: Dict[str, float]
    recommended_focus: List[str]
    llm_notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolutionProposal:
    strategy: str
    updated_hardware: Dict[str, Any]
    actions: List[str]
    retrieved_cases: List[Dict[str, Any]]
    rationale: str
    llm_notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass
class LLMConfig:
    provider: str = "mock"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    timeout: float = 600.0
    return_reasoning: bool = False
    retry_attempts: int = 2
    retry_temperature: float = 0.1


def _core_sort_key(core: str) -> Any:
    if isinstance(core, str) and core.startswith("core"):
        try:
            return int(core.replace("core", ""))
        except ValueError:
            return core
    return core
