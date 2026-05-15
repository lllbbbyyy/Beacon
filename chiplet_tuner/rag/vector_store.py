from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from chiplet_tuner.core.search_space import DEFAULT_HARDWARE_SEARCH_SPACE, infer_chip_size
from chiplet_tuner.rag.embeddings import HashingEmbeddingModel, Vector


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def hardware_fingerprint(hardware: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(hardware).encode("utf-8")).hexdigest()[:16]


def summarize_hardware(hardware: Dict[str, Any]) -> Dict[str, Any]:
    chiplets = hardware.get("chiplets", [])
    chiplet_types: Dict[str, int] = {}
    total_compute = 0.0
    total_buffer = 0.0
    for chip in chiplets:
        if not isinstance(chip, dict):
            continue
        chip_type = str(chip.get("type", "unknown"))
        chiplet_types[chip_type] = chiplet_types.get(chip_type, 0) + 1
        total_compute += float(chip.get("compute_units", 0.0))
        total_buffer += float(chip.get("buffer_size", 0.0))
    return {
        "fingerprint": hardware_fingerprint(hardware),
        "num_chiplets": hardware.get("num_chiplets", len(chiplets)),
        "chip_x": hardware.get("chip_x"),
        "chip_y": hardware.get("chip_y"),
        "dram_bw": hardware.get("dram_bw"),
        "nop_bw": hardware.get("nop_bw"),
        "micro_batch": hardware.get("micro_batch"),
        "tensor_parall": hardware.get("tensor_parall"),
        "chip_size": infer_chip_size(hardware, DEFAULT_HARDWARE_SEARCH_SPACE),
        "chiplet_types": chiplet_types,
        "total_compute_units": total_compute,
        "total_buffer_size": total_buffer,
    }


@dataclass
class HistoryRecord:
    record_id: str
    vector: Vector
    bottleneck_description: str
    hardware: Dict[str, Any]
    solution: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class RetrievedCase:
    score: float
    record: HistoryRecord


def build_retrieval_text(
    bottleneck_description: str,
    hardware: Dict[str, Any],
    extra_context: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "bottleneck_description": bottleneck_description,
        "hardware": hardware,
    }
    if extra_context:
        payload["extra_context"] = extra_context
    return canonical_json(payload)


class HistoryVectorStore:
    """JSON-backed vector database for previous tuning cases."""

    def __init__(
        self,
        path: Path | str,
        embedding_model: Optional[Any] = None,
        embedding_fn: Optional[Callable[[str], Vector]] = None,
    ) -> None:
        self.path = Path(path)
        self.embedding_model = embedding_model or HashingEmbeddingModel()
        self.embedding_fn = embedding_fn or self.embedding_model.embed
        self.records: List[HistoryRecord] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.records = []
            return
        with self.path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        self.records = [HistoryRecord(**item) for item in raw.get("records", [])]
        for record in self.records:
            record.metadata.setdefault("hardware_fingerprint", hardware_fingerprint(record.hardware))
            updated_hardware = record.solution.get("updated_hardware") if isinstance(record.solution, dict) else None
            if isinstance(updated_hardware, dict):
                record.metadata.setdefault("solution_hardware_fingerprint", hardware_fingerprint(updated_hardware))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": [asdict(record) for record in self.records]}
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def embed_state(
        self,
        bottleneck_description: str,
        hardware: Dict[str, Any],
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Vector:
        text = build_retrieval_text(bottleneck_description, hardware, extra_context)
        return self.embedding_fn(text)

    def add_case(
        self,
        bottleneck_description: str,
        hardware: Dict[str, Any],
        solution: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        record_id: Optional[str] = None,
        save: bool = True,
    ) -> HistoryRecord:
        metadata = dict(metadata or {})
        metadata.setdefault("hardware_fingerprint", hardware_fingerprint(hardware))
        updated_hardware = solution.get("updated_hardware") if isinstance(solution, dict) else None
        if isinstance(updated_hardware, dict):
            metadata.setdefault("solution_hardware_fingerprint", hardware_fingerprint(updated_hardware))
        vector = self.embed_state(bottleneck_description, hardware)
        if record_id is None:
            basis = canonical_json(
                {
                    "bottleneck_description": bottleneck_description,
                    "hardware": hardware,
                    "solution": solution,
                    "metrics": metrics or {},
                }
            )
            record_id = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]
        for existing in self.records:
            if existing.record_id == record_id:
                return existing
        record = HistoryRecord(
            record_id=record_id,
            vector=vector,
            bottleneck_description=bottleneck_description,
            hardware=hardware,
            solution=solution,
            metrics=metrics or {},
            metadata=metadata,
        )
        self.records.append(record)
        if save:
            self.save()
        return record

    def search(
        self,
        bottleneck_description: str,
        hardware: Dict[str, Any],
        top_k: int = 5,
        min_score: float = -1.0,
        metadata_filter: Optional[Dict[str, Any]] = None,
        exclude_hardware_fingerprints: Optional[set[str]] = None,
    ) -> List[RetrievedCase]:
        if top_k <= 0 or not self.records:
            return []
        query_vector = self.embed_state(bottleneck_description, hardware)
        hits = []
        for record in self.records:
            if metadata_filter and not self._metadata_matches(record.metadata, metadata_filter):
                continue
            if exclude_hardware_fingerprints and record.metadata.get(
                "hardware_fingerprint"
            ) in exclude_hardware_fingerprints:
                continue
            try:
                score = cosine_similarity(query_vector, record.vector)
            except ValueError:
                continue
            if score >= min_score:
                hits.append(RetrievedCase(score=score, record=record))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    def hardware_seen(self, hardware: Dict[str, Any]) -> bool:
        fingerprint = hardware_fingerprint(hardware)
        return any(record.metadata.get("hardware_fingerprint") == fingerprint for record in self.records)

    def all_hardware_fingerprints(self) -> set[str]:
        return {
            str(record.metadata["hardware_fingerprint"])
            for record in self.records
            if record.metadata.get("hardware_fingerprint")
        }

    def _metadata_matches(self, metadata: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        for key, value in expected.items():
            if isinstance(value, (list, tuple, set)):
                if metadata.get(key) not in value:
                    return False
            elif metadata.get(key) != value:
                return False
        return True

    def import_jsonl(self, path: Path | str, save: bool = True) -> int:
        imported = 0
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.add_case(
                    bottleneck_description=item["bottleneck_description"],
                    hardware=item["hardware"],
                    solution=item["solution"],
                    metrics=item.get("metrics"),
                    metadata=item.get("metadata"),
                    save=False,
                )
                imported += 1
        if save:
            self.save()
        return imported

    def __len__(self) -> int:
        return len(self.records)


def summarize_cases(cases: Iterable[RetrievedCase]) -> List[Dict[str, Any]]:
    return [
        {
            "score": round(case.score, 4),
            "bottleneck_description": case.record.bottleneck_description,
            "hardware_summary": summarize_hardware(case.record.hardware),
            "solution": case.record.solution,
            "metrics": case.record.metrics,
            "metadata": case.record.metadata,
        }
        for case in cases
    ]
