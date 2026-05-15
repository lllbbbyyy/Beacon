from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from chiplet_tuner.agents.react_graph import run_react_tool_graph
from chiplet_tuner.agents.tool_use import collect_generated_files, complete_json_continuing_messages, serialize_tool_results
from chiplet_tuner.core.schemas import BottleneckState, EvaluationResult, ModelAnalysisResult
from chiplet_tuner.llm.clients import LLMClient
from chiplet_tuner.llm.prompts import LAYER_LEVEL_ANALYSIS_PROMPT, LAYER_LEVEL_REACT_PROMPT, REACT_TOOL_USE_PROMPT
from chiplet_tuner.tools.analysis_tools import AnalysisToolbox, ToolContext


class LayerLevelAgent:
    """LLM-driven layer-level bottleneck diagnosis agent."""

    def __init__(self, llm: LLMClient, toolbox: AnalysisToolbox) -> None:
        self.llm = llm
        self.toolbox = toolbox

    def analyze(
        self,
        model_result: ModelAnalysisResult,
        evaluation: EvaluationResult,
        output_dir: Path,
    ) -> BottleneckState:
        model_message = self._model_message(model_result)
        context = ToolContext(
            output_dir=output_dir,
            evaluation=evaluation,
            model_analysis=model_message,
            current_hardware=evaluation.hardware,
            active_base=model_result.analysis_base,
        )
        react_result = run_react_tool_graph(
            llm=self.llm,
            toolbox=self.toolbox,
            system_prompt=f"{LAYER_LEVEL_REACT_PROMPT}\n\n{REACT_TOOL_USE_PROMPT}",
            task="react_tool_use",
            agent_name="layer_level",
            agent_goal=(
                "Inspect candidate layers and diagnose per-layer latency/energy impact and "
                "compute, memory, communication, buffer, scheduling, or imbalance root causes."
            ),
            context_payload={
                "model_candidates": model_message,
                "candidate_layers": model_result.candidate_layers,
            },
            context=context,
            max_steps=15,
        )
        tool_results = react_result.tool_results
        react_trace = react_result.transcript
        result = complete_json_continuing_messages(
            self.llm,
            react_result.messages,
            {
                "task": "layer_level_analysis",
                "final_decision_required": True,
                "instruction": (
                    "Tool use is complete. Continue from the previous messages and produce the "
                    "final layer-level bottleneck state. Use the real tool observations already "
                    "present in this conversation, not a separately summarized context."
                ),
                "output_prompt": LAYER_LEVEL_ANALYSIS_PROMPT,
            },
            validate_response=self._parse_state,
        )
        state = self._parse_state(result)
        state.llm_notes["tool_results"] = serialize_tool_results(tool_results)
        state.llm_notes["generated_files"] = collect_generated_files(tool_results)
        state.llm_notes["react_trace"] = react_trace
        state.llm_notes["react_backend"] = "langgraph"
        state.llm_notes["analysis_base"] = model_result.analysis_base
        return state

    def _model_message(self, model_result: ModelAnalysisResult) -> Dict[str, Any]:
        return {
            "metrics": model_result.metrics,
            "summary": model_result.summary,
            "global_findings": model_result.global_findings,
            "selected_views": model_result.selected_views,
            "candidate_layers": model_result.candidate_layers,
            "analysis_base": model_result.analysis_base,
        }

    def _parse_state(self, result: Dict[str, Any]) -> BottleneckState:
        primary_impact = self._require_choice(
            result,
            "primary_impact",
            {"latency", "energy", "monetary_cost", "mixed", "unknown"},
        )
        dominant_root_cause = self._require_choice(
            result,
            "dominant_root_cause",
            {"compute", "memory", "communication", "buffer", "scheduling", "imbalance", "mixed", "unknown"},
        )
        layer_diagnoses = self._normalize_layer_diagnoses(self._require_list(result, "layer_diagnoses"))
        retrieval_description = self._require_str(result, "retrieval_description")
        root_cause_summary = self._require_ratios(result, "root_cause_summary")
        recommended_focus = self._require_string_list(result, "recommended_focus")
        raw_notes = result.get("notes", {})
        notes = raw_notes if isinstance(raw_notes, dict) else {"raw_notes": raw_notes}
        return BottleneckState(
            primary_impact=primary_impact,
            dominant_root_cause=dominant_root_cause,
            layer_diagnoses=layer_diagnoses,
            retrieval_description=retrieval_description,
            root_cause_summary=root_cause_summary,
            recommended_focus=recommended_focus,
            llm_notes=notes,
        )

    def _normalize_layer_diagnoses(self, raw_items: List[Any]) -> List[Dict[str, Any]]:
        diagnoses: List[Dict[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                raise ValueError(f"Layer diagnosis must be an object, got {item!r}")
            ratios = item.get("root_cause_ratios", item.get("ratios", {}))
            if not isinstance(ratios, dict):
                ratios = {}
            diagnoses.append(
                {
                    "layerID": item.get("layerID", item.get("layer_id")),
                    "layerName": item.get("layerName", item.get("layer_name")),
                    "operator_group": item.get("operator_group", item.get("group", "unknown")),
                    "impact_types": self._normalize_string_list(item.get("impact_types", [])),
                    "root_causes": self._normalize_string_list(item.get("root_causes", [])),
                    "dominant_root_cause": item.get(
                        "dominant_root_cause",
                        item.get("bottleneck_type", "unknown"),
                    ),
                    "root_cause_ratios": {
                        key: self._optional_float(ratios.get(key, 0.0))
                        for key in ["compute", "memory", "communication", "buffer"]
                    },
                    "load_features": item.get("load_features", {}),
                    "diagnosis": item.get("diagnosis", item.get("problem_location", "")),
                }
            )
        if not diagnoses:
            raise ValueError("Layer-level result must include at least one layer diagnosis.")
        return diagnoses

    def _require_choice(self, payload: Dict[str, Any], key: str, choices: set[str]) -> str:
        value = self._require_str(payload, key)
        if value not in choices:
            raise ValueError(f"Invalid {key}: {value!r}; expected one of {sorted(choices)}")
        return value

    def _normalize_string_list(self, value: Any) -> List[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if isinstance(item, str) and item.strip()]

    def _optional_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _require_str(self, payload: Dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Layer-level LLM result must contain non-empty string field {key!r}")
        return value

    def _require_list(self, payload: Dict[str, Any], key: str) -> List[Any]:
        value = payload.get(key)
        if not isinstance(value, list):
            raise ValueError(f"Layer-level LLM result must contain list field {key!r}")
        return value

    def _require_string_list(self, payload: Dict[str, Any], key: str) -> List[str]:
        value = self._require_list(payload, key)
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError(f"Layer-level LLM result field {key!r} must be a list of non-empty strings")
        return value

    def _require_ratios(self, payload: Dict[str, Any], key: str) -> Dict[str, float]:
        value = payload.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"Layer-level LLM result must contain dict field {key!r}")
        missing = {"compute", "memory", "communication", "buffer"} - set(value)
        if missing:
            raise ValueError(f"Layer-level LLM result field {key!r} missing dimensions: {sorted(missing)}")
        ratios = {}
        for dim in ["compute", "memory", "communication", "buffer"]:
            try:
                ratios[dim] = float(value[dim])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Layer-level LLM result field {key!r}.{dim} must be numeric, got {value[dim]!r}"
                ) from exc
        return ratios
