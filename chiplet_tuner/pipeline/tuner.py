from __future__ import annotations

import copy
import json
import math
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chiplet_tuner.agents.layer_level import LayerLevelAgent
from chiplet_tuner.agents.model_level import ModelLevelAgent
from chiplet_tuner.agents.solution import SolutionGenerationAgent
from chiplet_tuner.core.io import write_json
from chiplet_tuner.core.progress import ProgressReporter
from chiplet_tuner.core.search_space import infer_chip_size
from chiplet_tuner.core.schemas import BottleneckState, EvaluationResult, ModelAnalysisResult, SolutionProposal
from chiplet_tuner.llm.clients import LLMClient
from chiplet_tuner.llm.tracing import LLMTraceRecorder
from chiplet_tuner.rag.vector_store import HistoryVectorStore, hardware_fingerprint
from chiplet_tuner.simulators.base import SimulatorAdapter


class MultiAgentTuner:
    """Orchestrates evaluation, LLM agents, RAG, and iterative tuning."""

    def __init__(
        self,
        model_agent: ModelLevelAgent,
        layer_agent: LayerLevelAgent,
        solution_agent: SolutionGenerationAgent,
        history_store: HistoryVectorStore,
        output_root: Path,
        simulator: Optional[SimulatorAdapter] = None,
        progress: Optional[ProgressReporter] = None,
    ) -> None:
        self.model_agent = model_agent
        self.layer_agent = layer_agent
        self.solution_agent = solution_agent
        self.history_store = history_store
        self.output_root = output_root
        self.simulator = simulator
        self.progress = progress or ProgressReporter(enabled=False)
        self.timing_events: List[Dict[str, Any]] = []

    def analyze_evaluation(
        self,
        evaluation: EvaluationResult,
        iteration: int = 0,
        output_dir: Optional[Path] = None,
        forbidden_hardware_fingerprints: Optional[set[str]] = None,
        total_iterations: Optional[int] = None,
        stage_offset: int = 0,
        total_stages: int = 4,
        search_state: Optional[Dict[str, Any]] = None,
        evaluation_bases: Optional[Dict[str, EvaluationResult]] = None,
    ) -> Tuple[ModelAnalysisResult, BottleneckState, SolutionProposal]:
        output_dir = output_dir or evaluation.run_dir
        evaluation_bases = evaluation_bases or {"current": evaluation}
        self._configure_llm_trace(output_dir)
        self._validate_required_metrics(evaluation.metrics)
        started = time.monotonic()
        try:
            with self.progress.task(
                iteration=iteration,
                total_iterations=total_iterations,
                stage=stage_offset + 1,
                total_stages=total_stages,
                component="agent:model_level",
                action="analyze model-level bottlenecks",
            ):
                model_result = self.model_agent.analyze(
                    evaluation,
                    output_dir,
                    search_state=search_state,
                    evaluation_bases=evaluation_bases,
                )
        finally:
            self._record_timing(
                iteration=iteration,
                component="agent:model_level",
                action="analyze model-level bottlenecks",
                duration_s=time.monotonic() - started,
            )
        analysis_evaluation = self._evaluation_for_analysis_base(model_result, evaluation_bases, default=evaluation)
        started = time.monotonic()
        try:
            with self.progress.task(
                iteration=iteration,
                total_iterations=total_iterations,
                stage=stage_offset + 2,
                total_stages=total_stages,
                component="agent:layer_level",
                action="diagnose candidate layers",
            ):
                state = self.layer_agent.analyze(model_result, analysis_evaluation, output_dir)
        finally:
            self._record_timing(
                iteration=iteration,
                component="agent:layer_level",
                action="diagnose candidate layers",
                duration_s=time.monotonic() - started,
            )
        started = time.monotonic()
        try:
            with self.progress.task(
                iteration=iteration,
                total_iterations=total_iterations,
                stage=stage_offset + 3,
                total_stages=total_stages,
                component="agent:solution_generation",
                action="retrieve history and propose hardware update",
            ):
                proposal = self.solution_agent.propose(
                    state,
                    analysis_evaluation.hardware,
                    output_dir=output_dir,
                    evaluation=analysis_evaluation,
                    forbidden_hardware_fingerprints=forbidden_hardware_fingerprints,
                )
        finally:
            self._record_timing(
                iteration=iteration,
                component="agent:solution_generation",
                action="retrieve history and propose hardware update",
                duration_s=time.monotonic() - started,
            )

        trace_path = output_dir / "analysis" / "agent_trace.json"
        started = time.monotonic()
        try:
            with self.progress.task(
                iteration=iteration,
                total_iterations=total_iterations,
                stage=stage_offset + 4,
                total_stages=total_stages,
                component="trace",
                action="write agent trace and LLM trace index",
            ):
                artifacts = self._collect_trace_artifacts(model_result, state, proposal, trace_path, output_dir)
                analysis_payload = self._build_agent_trace_payload(
                    iteration=iteration,
                    evaluation=analysis_evaluation,
                    model_result=model_result,
                    state=state,
                    proposal=proposal,
                    artifacts=artifacts,
                )
                write_json(trace_path, analysis_payload)
                model_result.generated_files["agent_trace_json"] = str(trace_path)
                self._write_llm_trace_index()
        finally:
            self._record_timing(
                iteration=iteration,
                component="trace",
                action="write agent trace and LLM trace index",
                duration_s=time.monotonic() - started,
            )
            self._write_timing_outputs()
        return model_result, state, proposal

    def _configure_llm_trace(self, output_dir: Path) -> None:
        traced_clients = [
            client
            for client in self._unique_llm_clients()
            if getattr(client, "trace_enabled", False) or getattr(client, "trace_recorder", None) is not None
        ]
        if not traced_clients:
            return
        trace_dir = output_dir / "llm_trace"
        recorder = None
        for client in traced_clients:
            current = getattr(client, "trace_recorder", None)
            if current is None:
                continue
            if Path(current.output_dir).resolve() == trace_dir.resolve():
                recorder = current
                break
        if recorder is None:
            recorder = LLMTraceRecorder(trace_dir)
        for client in traced_clients:
            client.trace_recorder = recorder

    def _unique_llm_clients(self) -> List[LLMClient]:
        clients: List[LLMClient] = []
        seen: set[int] = set()
        for agent in [self.model_agent, self.layer_agent, self.solution_agent]:
            client = getattr(agent, "llm", None)
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            clients.append(client)
        return clients

    def _evaluation_for_analysis_base(
        self,
        model_result: ModelAnalysisResult,
        evaluation_bases: Dict[str, EvaluationResult],
        default: EvaluationResult,
    ) -> EvaluationResult:
        base = model_result.analysis_base or {}
        key = base.get("base_key")
        if isinstance(key, str) and key in evaluation_bases:
            return evaluation_bases[key]
        iteration = base.get("iteration")
        if iteration is not None:
            iter_key = f"iter_{int(iteration):03d}"
            if iter_key in evaluation_bases:
                return evaluation_bases[iter_key]
        source = base.get("source")
        if isinstance(source, str) and source in evaluation_bases:
            return evaluation_bases[source]
        return default

    def _write_llm_trace_index(self) -> None:
        trace_entries: List[Dict[str, Any]] = []
        for trace_dir in sorted(self.output_root.glob("*/llm_trace")):
            calls_path = trace_dir / "prompt_calls.json"
            if not calls_path.exists():
                continue
            try:
                calls = json.loads(calls_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(calls, list):
                continue
            statuses: Dict[str, int] = {}
            duration_s = 0.0
            for call in calls:
                if not isinstance(call, dict):
                    continue
                status = str(call.get("status", "unknown"))
                statuses[status] = statuses.get(status, 0) + 1
                duration_s += self._duration_value(call.get("duration_s"))
            trace_entries.append(
                {
                    "run_dir": trace_dir.parent.name,
                    "trace_dir": self._relative_path(trace_dir, self.output_root),
                    "call_count": len(calls),
                    "duration_s": round(duration_s, 6),
                    "statuses": statuses,
                    "first_call": calls[0].get("display_index") if calls and isinstance(calls[0], dict) else None,
                    "last_call": calls[-1].get("display_index") if calls and isinstance(calls[-1], dict) else None,
                }
            )
        if trace_entries:
            write_json(self.output_root / "llm_trace_index.json", {"traces": trace_entries})

    def _record_timing(
        self,
        iteration: Optional[int],
        component: str,
        action: str,
        duration_s: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        event: Dict[str, Any] = {
            "iteration": iteration,
            "component": component,
            "action": action,
            "duration_s": round(float(duration_s), 6),
        }
        if metadata:
            event["metadata"] = metadata
        self.timing_events.append(event)

    def _iteration_timing_summary(self, iteration: int) -> Dict[str, Any]:
        events = [event for event in self.timing_events if event.get("iteration") == iteration]
        by_component = self._summarize_timing_events(events)
        return {
            "pipeline_event_count": len(events),
            "pipeline_total_s": round(sum(float(event.get("duration_s", 0.0)) for event in events), 6),
            "pipeline_by_component": by_component,
            "pipeline_events": events,
        }

    def _write_timing_outputs(self) -> Dict[str, Any]:
        summary = self._build_timing_summary()
        write_json(self.output_root / "timing_summary.json", summary)
        self._write_timing_markdown(summary, self.output_root / "timing_summary.md")
        return summary

    def _build_timing_summary(self) -> Dict[str, Any]:
        events = list(self.timing_events)
        by_iteration: Dict[str, Any] = {}
        for event in events:
            iteration = event.get("iteration")
            key = "none" if iteration is None else f"iter_{int(iteration):03d}"
            bucket = by_iteration.setdefault(
                key,
                {
                    "iteration": iteration,
                    "pipeline_event_count": 0,
                    "pipeline_total_s": 0.0,
                    "pipeline_by_component": {},
                    "pipeline_events": [],
                },
            )
            duration = float(event.get("duration_s", 0.0))
            bucket["pipeline_event_count"] += 1
            bucket["pipeline_total_s"] += duration
            bucket["pipeline_events"].append(event)
        for bucket in by_iteration.values():
            bucket["pipeline_total_s"] = round(float(bucket["pipeline_total_s"]), 6)
            bucket["pipeline_by_component"] = self._summarize_timing_events(bucket["pipeline_events"])

        return {
            "schema_version": 1,
            "description": (
                "Pipeline timings are local wall-clock durations. LLM API timings are measured per "
                "chat-completion call from request start to response/exception handling, so they include "
                "network latency and provider-side generation time."
            ),
            "pipeline": {
                "event_count": len(events),
                "total_recorded_s": round(sum(float(event.get("duration_s", 0.0)) for event in events), 6),
                "by_component": self._summarize_timing_events(events),
                "by_iteration": by_iteration,
                "events": events,
            },
            "llm_api": self._llm_trace_timing_summary(),
        }

    def _summarize_timing_events(self, events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for event in events:
            component = str(event.get("component", "unknown"))
            duration = float(event.get("duration_s", 0.0))
            bucket = summary.setdefault(
                component,
                {
                    "count": 0,
                    "total_s": 0.0,
                    "avg_s": 0.0,
                    "max_s": 0.0,
                },
            )
            bucket["count"] += 1
            bucket["total_s"] += duration
            bucket["max_s"] = max(float(bucket["max_s"]), duration)
        for bucket in summary.values():
            count = int(bucket["count"])
            bucket["total_s"] = round(float(bucket["total_s"]), 6)
            bucket["avg_s"] = round(float(bucket["total_s"]) / count, 6) if count else 0.0
            bucket["max_s"] = round(float(bucket["max_s"]), 6)
        return dict(sorted(summary.items()))

    def _llm_trace_timing_summary(self) -> Dict[str, Any]:
        traces: List[Dict[str, Any]] = []
        by_agent: Dict[str, Dict[str, Any]] = {}
        by_task: Dict[str, Dict[str, Any]] = {}
        total = self._empty_llm_timing_bucket()

        for trace_dir in sorted(self.output_root.glob("*/llm_trace")):
            calls_path = trace_dir / "prompt_calls.json"
            if not calls_path.exists():
                continue
            try:
                calls = json.loads(calls_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(calls, list):
                continue
            trace_bucket = self._empty_llm_timing_bucket()
            statuses: Dict[str, int] = {}
            for call in calls:
                if not isinstance(call, dict):
                    continue
                status = str(call.get("status", "unknown"))
                statuses[status] = statuses.get(status, 0) + 1
                agent = str(call.get("agent_name", "unknown"))
                task = str(call.get("task", "unknown"))
                duration = self._duration_value(call.get("duration_s"))
                usage = call.get("usage") if isinstance(call.get("usage"), dict) else {}
                self._accumulate_llm_call(total, duration, usage)
                self._accumulate_llm_call(trace_bucket, duration, usage)
                self._accumulate_llm_call(
                    by_agent.setdefault(agent, self._empty_llm_timing_bucket()),
                    duration,
                    usage,
                )
                self._accumulate_llm_call(
                    by_task.setdefault(task, self._empty_llm_timing_bucket()),
                    duration,
                    usage,
                )
            traces.append(
                {
                    "run_dir": trace_dir.parent.name,
                    "trace_dir": self._relative_path(trace_dir, self.output_root),
                    "statuses": statuses,
                    **self._finalize_llm_timing_bucket(trace_bucket),
                }
            )

        return {
            "total": self._finalize_llm_timing_bucket(total),
            "by_agent": {
                key: self._finalize_llm_timing_bucket(value)
                for key, value in sorted(by_agent.items())
            },
            "by_task": {
                key: self._finalize_llm_timing_bucket(value)
                for key, value in sorted(by_task.items())
            },
            "traces": traces,
        }

    def _empty_llm_timing_bucket(self) -> Dict[str, Any]:
        return {
            "call_count": 0,
            "duration_s": 0.0,
            "avg_duration_s": 0.0,
            "max_duration_s": 0.0,
            "usage": {},
        }

    def _accumulate_llm_call(self, bucket: Dict[str, Any], duration_s: float, usage: Dict[str, Any]) -> None:
        bucket["call_count"] += 1
        bucket["duration_s"] += duration_s
        bucket["max_duration_s"] = max(float(bucket["max_duration_s"]), duration_s)
        self._accumulate_numeric_tree(bucket["usage"], usage)

    def _finalize_llm_timing_bucket(self, bucket: Dict[str, Any]) -> Dict[str, Any]:
        count = int(bucket.get("call_count", 0))
        duration = round(float(bucket.get("duration_s", 0.0)), 6)
        return {
            "call_count": count,
            "duration_s": duration,
            "avg_duration_s": round(duration / count, 6) if count else 0.0,
            "max_duration_s": round(float(bucket.get("max_duration_s", 0.0)), 6),
            "usage": bucket.get("usage", {}),
        }

    def _accumulate_numeric_tree(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                target[key] = target.get(key, 0) + value
            elif isinstance(value, dict):
                child = target.setdefault(key, {})
                if isinstance(child, dict):
                    self._accumulate_numeric_tree(child, value)

    def _duration_value(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return number if math.isfinite(number) and number >= 0.0 else 0.0

    def _write_timing_markdown(self, summary: Dict[str, Any], path: Path) -> None:
        pipeline = summary.get("pipeline", {})
        llm_api = summary.get("llm_api", {})
        lines = [
            "# Timing Summary",
            "",
            "Pipeline timings are local wall-clock durations. LLM API timings are measured per request and include network latency plus provider-side generation time.",
            "",
            "## Pipeline By Component",
            "",
            "| component | calls | total_s | avg_s | max_s |",
            "|---|---:|---:|---:|---:|",
        ]
        by_component = pipeline.get("by_component", {}) if isinstance(pipeline, dict) else {}
        if isinstance(by_component, dict) and by_component:
            for component, bucket in by_component.items():
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(component),
                            str(bucket.get("count", 0)),
                            self._format_seconds(bucket.get("total_s")),
                            self._format_seconds(bucket.get("avg_s")),
                            self._format_seconds(bucket.get("max_s")),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("| N/A | 0 | N/A | N/A | N/A |")

        lines.extend(
            [
                "",
                "## LLM API By Agent",
                "",
                "| agent | calls | total_s | avg_s | max_s | prompt_tokens | cached_prompt_tokens | completion_tokens | reasoning_tokens | total_tokens |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        by_agent = llm_api.get("by_agent", {}) if isinstance(llm_api, dict) else {}
        if isinstance(by_agent, dict) and by_agent:
            for agent, bucket in by_agent.items():
                lines.append(self._llm_timing_table_row(agent, bucket))
        else:
            lines.append("| N/A | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")

        lines.extend(
            [
                "",
                "## LLM API By Run",
                "",
                "| run_dir | calls | total_s | avg_s | max_s | prompt_tokens | cached_prompt_tokens | completion_tokens | reasoning_tokens | total_tokens | statuses |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        traces = llm_api.get("traces", []) if isinstance(llm_api, dict) else []
        if isinstance(traces, list) and traces:
            for trace in traces:
                if not isinstance(trace, dict):
                    continue
                statuses = trace.get("statuses", {})
                lines.append(
                    self._llm_timing_table_row(
                        str(trace.get("run_dir", "unknown")),
                        trace,
                        extra=str(statuses) if isinstance(statuses, dict) else "",
                    )
                )
        else:
            lines.append("| N/A | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")

        lines.extend(
            [
                "",
                "## Pipeline Events",
                "",
                "| iter | component | action | duration_s |",
                "|---:|---|---|---:|",
            ]
        )
        events = pipeline.get("events", []) if isinstance(pipeline, dict) else []
        if isinstance(events, list) and events:
            for event in events:
                if not isinstance(event, dict):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(event.get("iteration")),
                            self._escape_markdown_table_cell(str(event.get("component", ""))),
                            self._escape_markdown_table_cell(str(event.get("action", ""))),
                            self._format_seconds(event.get("duration_s")),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("| N/A | N/A | N/A | N/A |")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _llm_timing_table_row(self, label: str, bucket: Dict[str, Any], extra: Optional[str] = None) -> str:
        usage = bucket.get("usage", {}) if isinstance(bucket.get("usage"), dict) else {}
        cells = [
            self._escape_markdown_table_cell(label),
            str(bucket.get("call_count", 0)),
            self._format_seconds(bucket.get("duration_s")),
            self._format_seconds(bucket.get("avg_duration_s")),
            self._format_seconds(bucket.get("max_duration_s")),
            self._format_integer(usage.get("prompt_tokens")),
            self._format_integer(self._nested_get(usage, ["prompt_tokens_details", "cached_tokens"])),
            self._format_integer(usage.get("completion_tokens")),
            self._format_integer(self._nested_get(usage, ["completion_tokens_details", "reasoning_tokens"])),
            self._format_integer(usage.get("total_tokens")),
        ]
        if extra is not None:
            cells.append(self._escape_markdown_table_cell(extra))
        return "| " + " | ".join(cells) + " |"

    def _nested_get(self, payload: Dict[str, Any], keys: List[str]) -> Any:
        current: Any = payload
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    def _format_seconds(self, value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"{number:.3f}"

    def _format_integer(self, value: Any) -> str:
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return "N/A"

    def _build_agent_trace_payload(
        self,
        iteration: int,
        evaluation: EvaluationResult,
        model_result: ModelAnalysisResult,
        state: BottleneckState,
        proposal: SolutionProposal,
        artifacts: Dict[str, str],
    ) -> Dict[str, Any]:
        return {
            "iteration": iteration,
            "evaluation": {
                "metrics": self._metrics_with_objective(evaluation.metrics),
            },
            "artifacts": artifacts,
            "model_analysis": self._compact_model_analysis(model_result),
            "bottleneck_state": self._compact_bottleneck_state(state),
            "solution_proposal": self._compact_solution_proposal(proposal),
            "llm_retry_summary": self._llm_retry_summary(),
            "timing": self._iteration_timing_summary(iteration),
        }

    def _collect_trace_artifacts(
        self,
        model_result: ModelAnalysisResult,
        state: BottleneckState,
        proposal: SolutionProposal,
        trace_path: Path,
        output_dir: Path,
    ) -> Dict[str, str]:
        artifacts: Dict[str, str] = {}
        for generated_files in [
            model_result.generated_files,
            state.llm_notes.get("generated_files", {}),
            proposal.llm_notes.get("generated_files", {}),
        ]:
            if isinstance(generated_files, dict):
                for key, value in generated_files.items():
                    artifacts.setdefault(str(key), self._relative_path(value, output_dir))
        llm_trace_dir = output_dir / "llm_trace"
        if llm_trace_dir.exists():
            artifacts["llm_trace_dir"] = self._relative_path(llm_trace_dir, output_dir)
        artifacts["agent_trace_json"] = self._relative_path(trace_path, output_dir)
        return artifacts

    def _relative_path(self, path: Any, base_dir: Path) -> str:
        candidate = Path(path)
        try:
            return str(candidate.resolve().relative_to(base_dir.resolve()))
        except (OSError, ValueError):
            return str(candidate)

    def _compact_model_analysis(self, model_result: ModelAnalysisResult) -> Dict[str, Any]:
        notes = model_result.llm_notes
        return {
            "summary": model_result.summary,
            "analysis_base": model_result.analysis_base,
            "candidate_layer_count": len(model_result.candidate_layers),
            "selected_bottleneck_objective": notes.get("selected_bottleneck_objective"),
            "selected_rank_views": model_result.selected_views,
            "critical_layers": notes.get("critical_layers", []),
            "global_findings": model_result.global_findings,
            "candidate_layers": [
                self._compact_layer_record(layer) for layer in model_result.candidate_layers
            ],
            "react_trace": self._compact_react_trace(notes.get("react_trace", [])),
            "notes": self._compact_notes(
                notes,
                exclude={
                    "critical_layers",
                    "selected_bottleneck_objective",
                    "selected_rank_view",
                },
            ),
        }

    def _compact_bottleneck_state(self, state: BottleneckState) -> Dict[str, Any]:
        return {
            "primary_impact": state.primary_impact,
            "dominant_root_cause": state.dominant_root_cause,
            "root_cause_summary": state.root_cause_summary,
            "recommended_focus": state.recommended_focus,
            "retrieval_description": state.retrieval_description,
            "layer_diagnosis_count": len(state.layer_diagnoses),
            "layer_diagnoses": [self._compact_layer_state(layer) for layer in state.layer_diagnoses],
            "react_trace": self._compact_react_trace(state.llm_notes.get("react_trace", [])),
            "notes": self._compact_notes(state.llm_notes),
        }

    def _compact_solution_proposal(self, proposal: SolutionProposal) -> Dict[str, Any]:
        notes = proposal.llm_notes
        return {
            "strategy": proposal.strategy,
            "actions": proposal.actions,
            "rationale": proposal.rationale,
            "updated_hardware": proposal.updated_hardware,
            "retrieved_case_count": len(proposal.retrieved_cases),
            "retrieved_cases": proposal.retrieved_cases,
            "updated_hardware_fingerprint": notes.get("updated_hardware_fingerprint"),
            "validation": notes.get("validation", []),
            "react_trace": self._compact_react_trace(notes.get("react_trace", [])),
            "notes": self._compact_notes(
                notes,
                exclude={"updated_hardware_fingerprint", "validation"},
            ),
        }

    def _compact_layer_record(self, layer: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "layer_id": layer.get("layer_id", layer.get("layerID")),
            "layer_name": layer.get("layer_name", layer.get("layerName")),
            "operator_group": layer.get("operator_group", layer.get("group")),
            "concern_types": layer.get("concern_types", []),
            "rank_metric": layer.get("rank_metric"),
            "rank_value": layer.get("rank_value"),
            "evidence": layer.get("evidence", []),
            "confidence": layer.get("confidence"),
        }

    def _compact_layer_state(self, layer_state: Dict[str, Any]) -> Dict[str, Any]:
        load_features = layer_state.get("load_features", {})
        if not isinstance(load_features, dict):
            load_features = {}
        return {
            "layerID": layer_state.get("layerID"),
            "layerName": layer_state.get("layerName"),
            "operator_group": layer_state.get("operator_group"),
            "impact_types": layer_state.get("impact_types", []),
            "root_causes": layer_state.get("root_causes", []),
            "dominant_root_cause": layer_state.get("dominant_root_cause"),
            "root_cause_ratios": layer_state.get("root_cause_ratios", {}),
            "load_features": {
                "occurrences": load_features.get("occurrences"),
                "cores": load_features.get("cores", []),
                "batches": load_features.get("batches", []),
                "latency_sum": load_features.get("latency_sum"),
                "latency_max": load_features.get("latency_max"),
                "critical_end": load_features.get("critical_end"),
                "energy": load_features.get("energy"),
                "features": self._compact_operator_features(load_features.get("features", {})),
            },
            "diagnosis": layer_state.get("diagnosis"),
        }

    def _compact_operator_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(features, dict):
            return {}
        return {
            key: features.get(key)
            for key in [
                "group",
                "head",
                "request",
                "tiling",
                "is_projection",
                "is_elementwise",
            ]
            if key in features
        }

    def _compact_react_trace(self, trace: Any) -> List[Dict[str, Any]]:
        if not isinstance(trace, list):
            return []
        compact_steps: List[Dict[str, Any]] = []
        keep_keys = [
            "step",
            "thought",
            "action",
            "tool_name",
            "arguments",
            "result_key",
            "finish_reason",
        ]
        for step in trace:
            if isinstance(step, dict):
                compact_steps.append({key: step[key] for key in keep_keys if key in step})
        return compact_steps

    def _compact_notes(
        self,
        notes: Dict[str, Any],
        exclude: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(notes, dict):
            return {}
        heavy_keys = {
            "analysis_summary",
            "generated_files",
            "heuristic_state_used",
            "react_trace",
            "tool_results",
        }
        excluded = heavy_keys | set(exclude or set())
        return {key: value for key, value in notes.items() if key not in excluded}

    def tune(
        self,
        initial_hardware: Dict[str, Any],
        iterations: int,
    ) -> Dict[str, Any]:
        if self.simulator is None:
            raise ValueError("A SimulatorAdapter is required for iterative tuning.")
        if iterations < 0:
            raise ValueError(f"iterations must be non-negative, got {iterations}")
        current_hardware = copy.deepcopy(initial_hardware)
        best: Optional[Dict[str, Any]] = None
        evaluated_fingerprints: set[str] = set()
        iteration_metrics: List[Dict[str, Any]] = []
        initial_objective: Optional[float] = None
        evaluations_by_iteration: Dict[int, EvaluationResult] = {}
        applied_changes_by_iteration: Dict[int, Dict[str, Any]] = {}
        baseline_stages = 1
        tuning_stages = 7

        baseline_fingerprint = hardware_fingerprint(current_hardware)
        started = time.monotonic()
        try:
            with self.progress.task(
                iteration=0,
                total_iterations=None,
                stage=1,
                total_stages=baseline_stages,
                component="evaluator",
                action="run initial baseline evaluation",
                detail=f"hardware_fingerprint={baseline_fingerprint}",
            ):
                baseline_evaluation = self.simulator.evaluate(current_hardware, 0)
        finally:
            self._record_timing(
                iteration=0,
                component="evaluator",
                action="run initial baseline evaluation",
                duration_s=time.monotonic() - started,
                metadata={"hardware_fingerprint": baseline_fingerprint},
            )

        evaluations_by_iteration[0] = baseline_evaluation
        evaluated_fingerprints.add(baseline_fingerprint)
        initial_objective = self._objective(baseline_evaluation.metrics)
        baseline_metrics = self._metrics_with_objective(baseline_evaluation.metrics)
        baseline_change = self._initial_change_summary()
        applied_changes_by_iteration[0] = baseline_change
        best = {
            "score": initial_objective,
            "hardware": copy.deepcopy(current_hardware),
            "metrics": baseline_metrics,
            "iteration": 0,
            "phase": "baseline",
        }
        write_json(self.output_root / "best_hardware.json", current_hardware)
        iteration_metrics.append(
            {
                "iteration": 0,
                "phase": "baseline",
                "metrics": baseline_metrics,
                "objective_change_vs_initial": 0.0,
                "objective_change_vs_previous_best": None,
                "is_new_best": True,
                "applied_change": baseline_change,
                "hardware_fingerprint": baseline_fingerprint,
            }
        )
        metrics_table_path = self._write_tuning_metrics_outputs(iteration_metrics)

        for iteration in range(1, iterations + 1):
            phase = "guided"
            latest_iteration = max(evaluations_by_iteration)
            latest_evaluation = evaluations_by_iteration[latest_iteration]
            latest_change = applied_changes_by_iteration.get(latest_iteration, self._unknown_change_summary())
            forbidden = evaluated_fingerprints | self.history_store.all_hardware_fingerprints()
            search_state, evaluation_bases = self._build_search_state(
                current_iteration=latest_iteration,
                evaluations_by_iteration=evaluations_by_iteration,
                applied_change=latest_change,
            )
            iteration_dir = self.output_root / f"iter_{iteration:03d}"
            model_result, state, proposal = self.analyze_evaluation(
                evaluation=latest_evaluation,
                iteration=iteration,
                output_dir=iteration_dir,
                forbidden_hardware_fingerprints=forbidden,
                total_iterations=iterations,
                stage_offset=0,
                total_stages=tuning_stages,
                search_state=search_state,
                evaluation_bases=evaluation_bases,
            )
            analysis_evaluation = self._evaluation_for_analysis_base(
                model_result,
                evaluation_bases,
                default=latest_evaluation,
            )
            analysis_fingerprint = hardware_fingerprint(analysis_evaluation.hardware)
            next_fingerprint = hardware_fingerprint(proposal.updated_hardware)
            if next_fingerprint in evaluated_fingerprints:
                started = time.monotonic()
                try:
                    with self.progress.task(
                        iteration=iteration,
                        total_iterations=iterations,
                        stage=5,
                        total_stages=tuning_stages,
                        component="pipeline",
                        action="repair duplicate hardware proposal",
                        detail=f"duplicate_fingerprint={next_fingerprint}",
                    ):
                        repaired, repair_actions = self.solution_agent.make_exploration_move(
                            proposal.updated_hardware,
                            state,
                            forbidden_hardware_fingerprints=forbidden,
                        )
                finally:
                    self._record_timing(
                        iteration=iteration,
                        component="pipeline",
                        action="repair duplicate hardware proposal",
                        duration_s=time.monotonic() - started,
                        metadata={"duplicate_fingerprint": next_fingerprint},
                    )
                if hardware_fingerprint(repaired) != next_fingerprint:
                    proposal.updated_hardware = repaired
                    proposal.actions.extend(action for action in repair_actions if action not in proposal.actions)
                    proposal.llm_notes.setdefault("validation", []).append(
                        "pipeline repaired duplicate next hardware after proposal"
                    )
                    next_fingerprint = hardware_fingerprint(repaired)

            pending_transition = self._build_pending_transition(
                iteration=iteration,
                phase=phase,
                evaluation=analysis_evaluation,
                model_result=model_result,
                state=state,
                proposal=proposal,
                current_hardware=analysis_evaluation.hardware,
                current_fingerprint=analysis_fingerprint,
                next_fingerprint=next_fingerprint,
            )
            write_json(self.output_root / "pending_transition.json", pending_transition)
            write_json(self.output_root / "latest_proposal.json", asdict(proposal))

            started = time.monotonic()
            try:
                with self.progress.task(
                    iteration=iteration,
                    total_iterations=iterations,
                    stage=6,
                    total_stages=tuning_stages,
                    component="evaluator",
                    action="evaluate proposed hardware",
                    detail=f"hardware_fingerprint={next_fingerprint}",
                ):
                    evaluation = self.simulator.evaluate(proposal.updated_hardware, iteration)
            finally:
                self._record_timing(
                    iteration=iteration,
                    component="evaluator",
                    action="evaluate proposed hardware",
                    duration_s=time.monotonic() - started,
                    metadata={"hardware_fingerprint": next_fingerprint},
                )

            started = time.monotonic()
            try:
                with self.progress.task(
                    iteration=iteration,
                    total_iterations=iterations,
                    stage=7,
                    total_stages=tuning_stages,
                    component="rag",
                    action="commit evaluated transition to history database",
                ):
                    self._commit_evaluated_transition(pending_transition, evaluation, iteration)
            finally:
                self._record_timing(
                    iteration=iteration,
                    component="rag",
                    action="commit evaluated transition to history database",
                    duration_s=time.monotonic() - started,
                )

            current_hardware = proposal.updated_hardware
            evaluated_fingerprints.add(next_fingerprint)
            evaluations_by_iteration[iteration] = evaluation
            applied_change = self._change_summary_from_transition(pending_transition)
            applied_changes_by_iteration[iteration] = applied_change
            score = self._objective(evaluation.metrics)
            previous_best_score = float(best["score"]) if best is not None else None
            is_new_best = previous_best_score is None or score < previous_best_score
            metrics = self._metrics_with_objective(evaluation.metrics)
            iteration_metrics.append(
                {
                    "iteration": iteration,
                    "phase": phase,
                    "metrics": metrics,
                    "objective_change_vs_initial": self._change_ratio(initial_objective, score),
                    "objective_change_vs_previous_best": (
                        None if previous_best_score is None else self._change_ratio(previous_best_score, score)
                    ),
                    "is_new_best": is_new_best,
                    "applied_change": applied_change,
                    "hardware_fingerprint": next_fingerprint,
                }
            )
            metrics_table_path = self._write_tuning_metrics_outputs(iteration_metrics)
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "hardware": copy.deepcopy(current_hardware),
                    "metrics": metrics,
                    "iteration": iteration,
                    "phase": phase,
                }
                write_json(self.output_root / "best_hardware.json", current_hardware)
            write_json(
                self.output_root / "pending_transition.json",
                {
                    "status": "evaluated",
                    "source_iteration": pending_transition.get("source_iteration"),
                    "after_iteration": iteration,
                    "hardware_fingerprint": next_fingerprint,
                    "before_metrics": pending_transition.get("before_metrics"),
                    "after_metrics": metrics,
                    "applied_change": applied_change,
                },
            )
            self.progress.info(
                iteration=iteration,
                total_iterations=iterations,
                component="iteration",
                message=(
                    f"DONE phase={phase} bottleneck={state.primary_impact}/{state.dominant_root_cause} "
                    f"actions={', '.join(proposal.actions)}"
                ),
            )

        metrics_table_path = self._write_tuning_metrics_outputs(iteration_metrics)
        timing_summary = self._write_timing_outputs()
        final = {
            "best": best or {},
            "next_hardware": current_hardware,
            "evaluated_designs": len(evaluated_fingerprints),
            "iterations": iterations,
            "tuning_iterations": iterations,
            "proposal_iterations": iterations,
            "evaluated_runs": len(iteration_metrics),
            "history_records": len(self.history_store),
            "llm_retry_summary": self._llm_retry_summary(),
            "metrics_table": str(metrics_table_path),
            "iteration_metrics": iteration_metrics,
            "timing_summary": timing_summary,
            "timing_summary_json": str(self.output_root / "timing_summary.json"),
            "timing_summary_md": str(self.output_root / "timing_summary.md"),
        }
        write_json(self.output_root / "tuning_summary.json", final)
        return final

    def _initial_change_summary(self) -> Dict[str, Any]:
        return {
            "source_iteration": None,
            "strategy": "initial_hardware",
            "actions": [],
            "summary": "initial hardware",
        }

    def _unknown_change_summary(self) -> Dict[str, Any]:
        return {
            "source_iteration": None,
            "strategy": "unknown",
            "actions": [],
            "summary": "no matching pending transition",
        }

    def _build_search_state(
        self,
        current_iteration: int,
        evaluations_by_iteration: Dict[int, EvaluationResult],
        applied_change: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, EvaluationResult]]:
        search_space = self._search_space()
        current = evaluations_by_iteration[current_iteration]
        previous = evaluations_by_iteration.get(current_iteration - 1)
        best_iteration = min(
            evaluations_by_iteration,
            key=lambda item: self._objective(evaluations_by_iteration[item].metrics),
        )
        best = evaluations_by_iteration[best_iteration]

        bases: Dict[str, EvaluationResult] = {
            "current": current,
            "best": best,
            f"iter_{current_iteration:03d}": current,
            f"iter_{best_iteration:03d}": best,
        }
        if previous is not None:
            bases["previous"] = previous
            bases[f"iter_{current_iteration - 1:03d}"] = previous
        for iteration, evaluation in evaluations_by_iteration.items():
            bases.setdefault(f"iter_{iteration:03d}", evaluation)

        current_record = self._search_state_record("current", "current", current_iteration, current, search_space)
        best_record = self._search_state_record("best", "best", best_iteration, best, search_space)
        previous_record = (
            self._search_state_record("previous", "previous", current_iteration - 1, previous, search_space)
            if previous is not None
            else None
        )

        previous_objective = self._objective(previous.metrics) if previous is not None else None
        current_objective = self._objective(current.metrics)
        objective_change_vs_previous = (
            None if previous_objective is None else self._change_ratio(previous_objective, current_objective)
        )
        last_change_improved = None if objective_change_vs_previous is None else objective_change_vs_previous < 0.0
        suggested = ["current"]
        if best_iteration != current_iteration:
            suggested.append("best")
        if previous is not None and last_change_improved is False:
            suggested.append("previous")

        state: Dict[str, Any] = {
            "current": current_record,
            "best": best_record,
            "previous": previous_record,
            "bases": {
                key: self._search_state_record(
                    source=key if key in {"current", "previous", "best"} else "iteration",
                    base_key=key,
                    iteration=self._iteration_from_run_dir(evaluation.run_dir),
                    evaluation=evaluation,
                    search_space=search_space,
                )
                for key, evaluation in bases.items()
            },
            "last_applied_change": applied_change,
            "search_observation": {
                "current_is_best": best_iteration == current_iteration,
                "best_iteration": best_iteration,
                "objective_change_convention": "(after - before) / before; negative means objective decreased and improved",
                "objective_change_vs_previous": objective_change_vs_previous,
                "last_change_improved_objective": last_change_improved,
                "suggested_base_choices": suggested,
            },
            "available_bases": sorted(bases),
        }
        return state, bases

    def _search_state_record(
        self,
        source: str,
        base_key: str,
        iteration: Optional[int],
        evaluation: EvaluationResult,
        search_space: Dict[str, Any],
    ) -> Dict[str, Any]:
        metrics = self._metrics_with_objective(evaluation.metrics)
        return {
            "source": source,
            "base_key": base_key,
            "iteration": iteration,
            "run_dir": str(evaluation.run_dir),
            "metrics": metrics,
            "objective": metrics["objective"],
            "hardware_fingerprint": hardware_fingerprint(evaluation.hardware),
            "hardware_summary": self._compact_hardware_summary(evaluation.hardware, search_space),
        }

    def _iteration_from_run_dir(self, run_dir: Path) -> Optional[int]:
        name = run_dir.name
        if not name.startswith("iter_"):
            return None
        try:
            return int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            return None

    def _compact_hardware_summary(self, hardware: Dict[str, Any], search_space: Dict[str, Any]) -> Dict[str, Any]:
        chiplets = hardware.get("chiplets", [])
        chiplet_types: Dict[str, int] = {}
        compute_units: Dict[str, int] = {}
        buffer_sizes: Dict[str, int] = {}
        if isinstance(chiplets, list):
            chiplet_types = dict(Counter(str(chip.get("type")) for chip in chiplets if isinstance(chip, dict)))
            compute_units = dict(Counter(str(chip.get("compute_units")) for chip in chiplets if isinstance(chip, dict)))
            buffer_sizes = dict(Counter(str(chip.get("buffer_size")) for chip in chiplets if isinstance(chip, dict)))
        try:
            chip_size = infer_chip_size(hardware, search_space)
        except Exception:
            chip_size = None
        return {
            "chip_size": chip_size,
            "num_chiplets": hardware.get("num_chiplets"),
            "chip_x": hardware.get("chip_x"),
            "chip_y": hardware.get("chip_y"),
            "dram_bw": hardware.get("dram_bw"),
            "nop_bw": hardware.get("nop_bw"),
            "micro_batch": hardware.get("micro_batch"),
            "tensor_parall": hardware.get("tensor_parall"),
            "chiplet_types": chiplet_types,
            "compute_units": compute_units,
            "buffer_sizes": buffer_sizes,
            "chiplet_layout": self._compact_chiplet_layout(hardware),
            "layout_note": "summary includes type_sequence/type_grid; request full hardware if exact fields matter",
        }

    def _compact_chiplet_layout(self, hardware: Dict[str, Any]) -> Dict[str, Any]:
        chiplets = hardware.get("chiplets", [])
        if not isinstance(chiplets, list):
            chiplets = []
        type_sequence = [
            str(chip.get("type", "unknown")) if isinstance(chip, dict) else "unknown"
            for chip in chiplets
        ]
        chip_x = self._positive_int_or_none(hardware.get("chip_x"))
        chip_y = self._positive_int_or_none(hardware.get("chip_y"))
        type_grid: List[List[Optional[str]]] = []
        if chip_x and chip_y:
            for y in range(chip_y):
                row: List[Optional[str]] = []
                for x in range(chip_x):
                    idx = y * chip_x + x
                    row.append(type_sequence[idx] if idx < len(type_sequence) else None)
                type_grid.append(row)
        return {
            "layout_order_assumption": "row_major_y_then_x",
            "layout_complete": bool(chip_x and chip_y and len(type_sequence) == chip_x * chip_y),
            "type_sequence": type_sequence,
            "type_grid": type_grid,
        }

    def _positive_int_or_none(self, value: Any) -> Optional[int]:
        try:
            integer = int(value)
        except (TypeError, ValueError):
            return None
        return integer if integer > 0 else None

    def _change_summary_from_transition(self, transition: Dict[str, Any]) -> Dict[str, Any]:
        solution = transition.get("solution", {})
        if not isinstance(solution, dict):
            solution = {}
        actions = solution.get("actions", [])
        if not isinstance(actions, list):
            actions = []
        strategy = str(solution.get("strategy", "unknown"))
        return {
            "source_iteration": transition.get("source_iteration"),
            "strategy": strategy,
            "actions": [str(action) for action in actions],
            "summary": self._summarize_change(strategy, actions),
        }

    def _summarize_change(self, strategy: str, actions: List[Any], max_actions: int = 3) -> str:
        clean_actions = [str(action) for action in actions if str(action).strip()]
        if not clean_actions:
            return strategy
        head = clean_actions[:max_actions]
        suffix = f"; +{len(clean_actions) - max_actions} more" if len(clean_actions) > max_actions else ""
        return f"{strategy}: " + "; ".join(head) + suffix

    def _improvement_ratio(self, before: float, after: float) -> float:
        return (before - after) / before

    def _change_ratio(self, before: float, after: float) -> float:
        return (after - before) / before

    def _write_tuning_metrics_outputs(self, rows: List[Dict[str, Any]]) -> Path:
        table_path = self.output_root / "tuning_metrics_table.md"
        self._write_tuning_metrics_table(rows, table_path)
        write_json(
            self.output_root / "tuning_metrics.json",
            {
                "metrics_table": str(table_path),
                "completed_evaluations": len(rows),
                "latest_iteration": rows[-1].get("iteration") if rows else None,
                "rows": rows,
            },
        )
        return table_path

    def _write_tuning_metrics_table(self, rows: List[Dict[str, Any]], path: Path) -> None:
        lines = [
            "# Tuning Metrics",
            "",
            "Objective-change columns use (after - before) / before for objective = latency * energy * mc. "
            "Negative values mean the objective decreased and improved; positive values mean it increased and regressed.",
            "",
            "| Run | Latency | Energy | MC | Objective | Obj vs Initial | Obj vs Previous Best | New Best | Applied Change |",
            "|---|---:|---:|---:|---:|---:|---:|:---:|---|",
        ]
        for row in rows:
            metrics = row.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            applied_change = row.get("applied_change", {})
            if not isinstance(applied_change, dict):
                applied_change = {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        self._iteration_table_label(row),
                        self._format_number(metrics.get("latency")),
                        self._format_number(metrics.get("energy")),
                        self._format_number(metrics.get("mc")),
                        self._format_number(metrics.get("objective")),
                        self._format_percent(
                            row.get("objective_change_vs_initial", row.get("objective_improvement_vs_initial"))
                        ),
                        self._format_percent(
                            row.get(
                                "objective_change_vs_previous_best",
                                row.get("objective_improvement_vs_previous_best"),
                            )
                        ),
                        "yes" if row.get("is_new_best") else "no",
                        self._escape_markdown_table_cell(str(applied_change.get("summary", ""))),
                    ]
                )
                + " |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _iteration_table_label(self, row: Dict[str, Any]) -> str:
        iteration = row.get("iteration")
        phase = str(row.get("phase", ""))
        try:
            iteration_id = int(iteration)
        except (TypeError, ValueError):
            return "N/A"
        if iteration_id == 0 and phase != "final_validation":
            return "init"
        label = f"iter{iteration_id:03d}"
        if phase == "final_validation":
            return f"{label} final"
        return label

    def _format_number(self, value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"{number:.4g}"

    def _format_percent(self, value: Any) -> str:
        if value is None:
            return "N/A"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"{number * 100:+.2f}%"

    def _escape_markdown_table_cell(self, value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    def _llm_retry_summary(self) -> Dict[str, Any]:
        summaries = [client.retry_summary() for client in self._unique_llm_clients()]
        events: List[Dict[str, Any]] = []
        for summary in summaries:
            raw_events = summary.get("events", [])
            if isinstance(raw_events, list):
                events.extend(event for event in raw_events if isinstance(event, dict))
        by_task: Dict[str, int] = {}
        by_agent: Dict[str, int] = {}
        by_error_type: Dict[str, int] = {}
        retry_attempt_count = 0
        exhausted_count = 0
        for event in events:
            event_type = str(event.get("event", "retry"))
            if event_type == "retry":
                retry_attempt_count += 1
            if event_type == "retry_exhausted":
                exhausted_count += 1
            task = str(event.get("task", "unknown"))
            agent_name = str(event.get("agent_name", "unknown"))
            error_type = str(event.get("error_type", "unknown"))
            by_task[task] = by_task.get(task, 0) + 1
            by_agent[agent_name] = by_agent.get(agent_name, 0) + 1
            by_error_type[error_type] = by_error_type.get(error_type, 0) + 1
        return {
            "event_count": len(events),
            "retry_attempt_count": retry_attempt_count,
            "exhausted_count": exhausted_count,
            "by_task": by_task,
            "by_agent": by_agent,
            "by_error_type": by_error_type,
            "events": events,
        }

    def _build_pending_transition(
        self,
        iteration: int,
        phase: str,
        evaluation: EvaluationResult,
        model_result: ModelAnalysisResult,
        state: BottleneckState,
        proposal: SolutionProposal,
        current_hardware: Dict[str, Any],
        current_fingerprint: str,
        next_fingerprint: str,
    ) -> Dict[str, Any]:
        return {
            "source_iteration": iteration,
            "phase": phase,
            "analysis_base": model_result.analysis_base,
            "bottleneck_description": state.retrieval_description,
            "bottleneck_state": {
                "primary_impact": state.primary_impact,
                "dominant_root_cause": state.dominant_root_cause,
                "root_cause_summary": state.root_cause_summary,
                "recommended_focus": state.recommended_focus,
                "layer_diagnoses": state.layer_diagnoses,
            },
            "model_summary": model_result.summary,
            "candidate_layers": model_result.candidate_layers,
            "hardware": copy.deepcopy(current_hardware),
            "hardware_fingerprint": current_fingerprint,
            "before_metrics": self._metrics_with_objective(evaluation.metrics),
            "solution": {
                "strategy": proposal.strategy,
                "actions": proposal.actions,
                "rationale": proposal.rationale,
                "updated_hardware": proposal.updated_hardware,
                "updated_hardware_fingerprint": next_fingerprint,
            },
            "next_hardware_fingerprint": next_fingerprint,
        }

    def _commit_evaluated_transition(
        self,
        pending_transition: Dict[str, Any],
        after_evaluation: EvaluationResult,
        after_iteration: int,
    ) -> None:
        before_metrics = pending_transition["before_metrics"]
        after_metrics = self._metrics_with_objective(after_evaluation.metrics)
        improvement = self._compare_metrics(before_metrics, after_metrics)
        solution = dict(pending_transition["solution"])
        solution.pop("updated_hardware_fingerprint", None)
        self.history_store.add_case(
            bottleneck_description=pending_transition["bottleneck_description"],
            bottleneck_state=pending_transition["bottleneck_state"],
            hardware=pending_transition["hardware"],
            solution=solution,
            metrics={
                "before": before_metrics,
                "after": after_metrics,
                "improvement": improvement,
            },
            metadata={
                "source_iteration": pending_transition["source_iteration"],
                "after_iteration": after_iteration,
                "phase": pending_transition["phase"],
                "analysis_base": pending_transition.get("analysis_base", {}),
                "base_iteration": (pending_transition.get("analysis_base") or {}).get("iteration"),
                "hardware_fingerprint": pending_transition["hardware_fingerprint"],
                "next_hardware_fingerprint": pending_transition["next_hardware_fingerprint"],
                "primary_impact": pending_transition["bottleneck_state"]["primary_impact"],
                "dominant_root_cause": pending_transition["bottleneck_state"]["dominant_root_cause"],
                "before_objective": before_metrics["objective"],
                "after_objective": after_metrics["objective"],
                "objective_change_ratio": improvement["objective_change_ratio"],
                "objective_improvement_ratio": improvement["objective_improvement_ratio"],
                "top_layers": [
                    layer.get("layer_name") or layer.get("layerName")
                    for layer in pending_transition.get("candidate_layers", [])[:5]
                ],
            },
        )

    def _compare_metrics(self, before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
        improvement: Dict[str, float] = {}
        for key in ["latency", "energy", "mc", "objective"]:
            before_value = self._required_positive_metric(before, key)
            after_value = self._required_positive_metric(after, key)
            improvement[f"{key}_before"] = before_value
            improvement[f"{key}_after"] = after_value
            improvement[f"{key}_ratio"] = after_value / before_value
            improvement[f"{key}_change_ratio"] = (after_value - before_value) / before_value
            improvement[f"{key}_improvement_ratio"] = (before_value - after_value) / before_value
        return improvement

    def _search_space(self) -> Dict[str, Any]:
        if self.simulator is None:
            raise ValueError("A SimulatorAdapter is required to resolve search space.")
        return self.simulator.schema().get("search_space", {})

    def _objective(self, metrics: Dict[str, float]) -> float:
        self._validate_required_metrics(metrics)
        latency = self._required_positive_metric(metrics, "latency")
        energy = self._required_positive_metric(metrics, "energy")
        mc = self._required_positive_metric(metrics, "mc")
        return latency * energy * mc

    def _metrics_with_objective(self, metrics: Dict[str, float]) -> Dict[str, float]:
        return {**metrics, "objective": self._objective(metrics)}

    def _validate_required_metrics(self, metrics: Dict[str, float]) -> None:
        for key in ["latency", "energy", "mc"]:
            self._required_positive_metric(metrics, key)

    def _required_positive_metric(self, metrics: Dict[str, float], key: str) -> float:
        if key not in metrics:
            raise KeyError(f"Evaluation metrics missing required key: {key}")
        try:
            value = float(metrics[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Evaluation metric {key} must be numeric, got {metrics[key]!r}") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Evaluation metric {key} must be finite and positive, got {value!r}")
        return value
