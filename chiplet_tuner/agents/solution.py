from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from chiplet_tuner.agents.react_graph import run_react_tool_graph
from chiplet_tuner.agents.tool_use import complete_json_continuing_messages, serialize_tool_results
from chiplet_tuner.core.schemas import BottleneckState, EvaluationResult, SolutionProposal
from chiplet_tuner.core.search_space import (
    DEFAULT_HARDWARE_SEARCH_SPACE,
    chip_type_candidates,
    infer_chip_size,
    materialize_hardware,
    normalize_hardware_to_design_space,
    step_chip_size,
    system_param_candidates,
)
from chiplet_tuner.core.utils import clamp_to_candidates
from chiplet_tuner.llm.clients import LLMClient
from chiplet_tuner.llm.prompts import REACT_TOOL_USE_PROMPT, SOLUTION_GENERATION_PROMPT, SOLUTION_REACT_PROMPT
from chiplet_tuner.rag.vector_store import HistoryVectorStore, hardware_fingerprint, summarize_cases
from chiplet_tuner.tools.analysis_tools import AnalysisToolbox, ToolContext


class SolutionGenerationAgent:
    """LLM-driven solution generator with RAG historical context."""

    def __init__(
        self,
        llm: LLMClient,
        store: HistoryVectorStore,
        toolbox: AnalysisToolbox,
        simulator_schema: Dict[str, Any],
        top_k: int = 5,
    ) -> None:
        self.llm = llm
        self.store = store
        self.toolbox = toolbox
        self.simulator_schema = simulator_schema
        self.top_k = top_k

    def propose(
        self,
        state: BottleneckState,
        hardware: Dict[str, Any],
        output_dir: Path,
        evaluation: Optional[EvaluationResult] = None,
        forbidden_hardware_fingerprints: Optional[Set[str]] = None,
    ) -> SolutionProposal:
        forbidden_hardware_fingerprints = set(forbidden_hardware_fingerprints or set())
        cases = self.store.search(state.retrieval_description, hardware, top_k=self.top_k)
        summarized_cases = summarize_cases(cases)
        state_message = self._state_message(state)
        context = ToolContext(
            output_dir=output_dir,
            evaluation=evaluation,
            bottleneck_state=state_message,
            current_hardware=hardware,
            retrieved_cases=summarized_cases,
            simulator_schema=self.simulator_schema,
            active_base=state.llm_notes.get("analysis_base", {}),
        )
        react_result = run_react_tool_graph(
            llm=self.llm,
            toolbox=self.toolbox,
            system_prompt=f"{SOLUTION_REACT_PROMPT}\n\n{REACT_TOOL_USE_PROMPT}",
            task="react_tool_use",
            agent_name="solution_generation",
            agent_goal=(
                "Gather hardware evidence and materialize candidate updates before proposing the next legal "
                "hardware configuration. Retrieved history is already provided in context."
            ),
            context_payload={
                "bottleneck_state": state_message,
                "analysis_base": state.llm_notes.get("analysis_base", {}),
                "current_hardware": hardware,
                "retrieved_cases": summarized_cases,
                "simulator_schema": self.simulator_schema,
                "search_space": self.simulator_schema.get("search_space", {}),
                "forbidden_hardware_fingerprints": sorted(forbidden_hardware_fingerprints),
            },
            context=context,
            max_steps=6,
        )
        tool_results = react_result.tool_results
        react_trace = react_result.transcript
        result = complete_json_continuing_messages(
            self.llm,
            react_result.messages,
            {
                "task": "solution_generation",
                "final_decision_required": True,
                "instruction": (
                    "Tool use is complete. Continue from the previous messages and produce the "
                    "final solution proposal. Use the real tool observations already present "
                    "in this conversation, not a separately summarized context."
                ),
                "output_prompt": SOLUTION_GENERATION_PROMPT,
            },
            validate_response=lambda payload: self._validate_solution_result(payload, tool_results),
        )
        strategy = self._require_str(result, "strategy")
        proposed_hardware, candidate_actions = self._proposed_hardware_update(result, tool_results)
        actions = self._require_string_list(result, "actions")
        for action in candidate_actions:
            if action not in actions:
                actions.append(action)
        rationale = self._require_str(result, "rationale", allow_empty=True)
        updated_hardware, validation_notes = self._normalize_hardware(
            proposed=proposed_hardware,
            current=hardware,
        )
        if hardware_fingerprint(updated_hardware) in forbidden_hardware_fingerprints:
            repaired, repair_actions = self.make_exploration_move(
                hardware=updated_hardware,
                state=state,
                forbidden_hardware_fingerprints=forbidden_hardware_fingerprints,
            )
            if hardware_fingerprint(repaired) != hardware_fingerprint(updated_hardware):
                updated_hardware = repaired
                validation_notes.append("proposal matched an evaluated hardware fingerprint; applied exploration repair")
                validation_notes.extend(repair_actions)
            else:
                validation_notes.append("proposal matched an evaluated hardware fingerprint and no legal repair was found")
        inferred_actions = self._diff_actions(hardware, updated_hardware)
        existing_action_prefixes = {action.split(":", 1)[0] for action in actions}
        for action in inferred_actions:
            prefix = action.split(":", 1)[0]
            if action not in actions and prefix not in existing_action_prefixes:
                actions.append(action)
                existing_action_prefixes.add(prefix)
        raw_notes = result.get("notes", {})
        notes = dict(raw_notes) if isinstance(raw_notes, dict) else {"raw_notes": raw_notes}
        notes["validation"] = validation_notes
        notes["updated_hardware_fingerprint"] = hardware_fingerprint(updated_hardware)
        notes["tool_results"] = serialize_tool_results(tool_results)
        notes["react_trace"] = react_trace
        notes["react_backend"] = "langgraph"
        return SolutionProposal(
            strategy=strategy,
            updated_hardware=updated_hardware,
            actions=actions,
            retrieved_cases=summarized_cases,
            rationale=rationale,
            llm_notes=notes,
        )

    def _validate_solution_result(
        self,
        result: Dict[str, Any],
        tool_results: Dict[str, Any],
    ) -> None:
        self._require_str(result, "strategy")
        self._proposed_hardware_update(result, tool_results)
        self._require_string_list(result, "actions")
        self._require_str(result, "rationale", allow_empty=True)

    def _proposed_hardware_update(
        self,
        result: Dict[str, Any],
        tool_results: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        if result.get("selected_hardware_candidate") or result.get("hardware_candidate_key"):
            selected_candidate = self._selected_tool_candidate(result, tool_results)
            if selected_candidate is not None:
                return selected_candidate
        hardware_update = result.get("hardware_update")
        if isinstance(hardware_update, dict):
            return hardware_update, []
        updated_hardware = result.get("updated_hardware")
        if isinstance(updated_hardware, dict):
            return updated_hardware, []
        selected_candidate = self._selected_tool_candidate(result, tool_results)
        if selected_candidate is not None:
            return selected_candidate
        raise ValueError(
            "Solution LLM result must contain object field 'hardware_update' or select a hardware candidate tool result."
        )

    def _selected_tool_candidate(
        self,
        result: Dict[str, Any],
        tool_results: Dict[str, Any],
    ) -> Optional[Tuple[Dict[str, Any], List[str]]]:
        selected_key = result.get("selected_hardware_candidate", result.get("hardware_candidate_key"))
        candidate_keys = [
            key
            for key, value in tool_results.items()
            if value.name in {"modify_hardware_parameter", "materialize_hardware_candidate", "step_hardware_parameter"}
        ]
        if isinstance(selected_key, str) and selected_key:
            candidate_keys = [selected_key] if selected_key in tool_results else candidate_keys
        if not candidate_keys:
            return None
        payload = tool_results[candidate_keys[-1]].payload
        candidate = payload.get("updated_hardware")
        if not isinstance(candidate, dict):
            return None
        raw_actions = payload.get("actions", [])
        actions = [str(action) for action in raw_actions if isinstance(action, str)]
        return candidate, actions

    def _state_message(self, state: BottleneckState) -> Dict[str, Any]:
        return {
            "primary_impact": state.primary_impact,
            "dominant_root_cause": state.dominant_root_cause,
            "layer_diagnoses": state.layer_diagnoses,
            "retrieval_description": state.retrieval_description,
            "root_cause_summary": state.root_cause_summary,
            "recommended_focus": state.recommended_focus,
        }

    def make_exploration_move(
        self,
        hardware: Dict[str, Any],
        state: BottleneckState,
        forbidden_hardware_fingerprints: Optional[Set[str]] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        forbidden = set(forbidden_hardware_fingerprints or set())
        focus_order = list(state.recommended_focus) or self._focus_for_bottleneck(state.dominant_root_cause)
        focus_order.extend(["chip_size", "dram_bw", "nop_bw", "tensor_parall", "micro_batch", "chiplet_type"])
        directions = self._directions_for_bottleneck(state.dominant_root_cause)
        tried = set()
        for key in focus_order:
            if key in tried:
                continue
            tried.add(key)
            for direction in directions.get(key, [1, -1]):
                candidate = copy.deepcopy(hardware)
                actions: List[str] = []
                if key in {"chip_size", "compute_units", "buffer_size"}:
                    self._step_compute_spec(candidate, direction, actions)
                elif key in {"chiplet_type", "type"}:
                    self._step_chiplet_type(candidate, direction, actions)
                elif key in candidate:
                    self._step_top_level(candidate, key, direction, actions)
                else:
                    self._step_chip_field(candidate, key, direction, actions)
                if actions and hardware_fingerprint(candidate) not in forbidden:
                    return candidate, actions
        return hardware, []

    def _normalize_hardware(
        self,
        proposed: Dict[str, Any],
        current: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        return normalize_hardware_to_design_space(
            proposed=proposed,
            current=current,
            search_space=self._search_space(),
        )

    def _require_dict(self, payload: Dict[str, Any], key: str) -> Dict[str, Any]:
        value = payload.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"Solution LLM result must contain object field {key!r}")
        return value

    def _require_str(self, payload: Dict[str, Any], key: str, allow_empty: bool = False) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or (not allow_empty and not value.strip()):
            raise ValueError(f"Solution LLM result must contain string field {key!r}")
        return value

    def _require_string_list(self, payload: Dict[str, Any], key: str) -> List[str]:
        value = payload.get(key)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"Solution LLM result field {key!r} must be a list of strings")
        return list(value)

    def _numeric_candidates(self, candidates: Sequence[Any]) -> List[float]:
        numeric: List[float] = []
        for item in candidates:
            if isinstance(item, bool):
                return []
            if not isinstance(item, (int, float)):
                return []
            numeric.append(float(item))
        return numeric

    def _step_top_level(self, hardware: Dict[str, Any], key: str, direction: int, actions: List[str]) -> None:
        search_space = self._search_space()
        candidates = system_param_candidates(search_space, key)
        if key not in hardware or not candidates or not self._numeric_candidates(candidates):
            return
        old = int(hardware[key])
        new = clamp_to_candidates(old, candidates, direction)
        if new != old:
            hardware[key] = new
            actions.append(f"{key}: {old}->{new}")

    def _step_chip_field(self, hardware: Dict[str, Any], key: str, direction: int, actions: List[str]) -> None:
        if key in {"compute_units", "buffer_size"}:
            self._step_compute_spec(hardware, direction, actions)
        elif key in {"type", "chiplet_type"}:
            self._step_chiplet_type(hardware, direction, actions)

    def _step_compute_spec(self, hardware: Dict[str, Any], direction: int, actions: List[str]) -> None:
        search_space = self._search_space()
        old_size = infer_chip_size(hardware, search_space)
        try:
            new_size = step_chip_size(hardware, search_space, direction)
        except ValueError:
            return
        if old_size == new_size:
            return
        old_num = hardware.get("num_chiplets")
        old_shape = (hardware.get("chip_y"), hardware.get("chip_x"))
        updated = materialize_hardware(
            template=hardware,
            chip_size=new_size,
            chiplet_types=[chip.get("type") for chip in hardware.get("chiplets", []) if isinstance(chip, dict)],
            search_space=search_space,
        )
        hardware.clear()
        hardware.update(updated)
        actions.append(
            f"chip_size: {old_size}->{new_size}; "
            f"shape {old_shape[0]}x{old_shape[1]}->{hardware.get('chip_y')}x{hardware.get('chip_x')}; "
            f"num_chiplets {old_num}->{hardware.get('num_chiplets')}"
        )

    def _step_chiplet_type(self, hardware: Dict[str, Any], direction: int, actions: List[str]) -> None:
        candidates = chip_type_candidates(self._search_space())
        if len(candidates) < 2:
            return
        chiplets = hardware.get("chiplets", [])
        if not isinstance(chiplets, list):
            return
        for idx, chip in enumerate(chiplets):
            if not isinstance(chip, dict):
                continue
            old = str(chip.get("type", candidates[0]))
            old_idx = candidates.index(old) if old in candidates else 0
            new_idx = max(0, min(len(candidates) - 1, old_idx + direction))
            if new_idx == old_idx:
                continue
            chip["type"] = candidates[new_idx]
            actions.append(f"chiplet_type[{idx}]: {old}->{chip['type']}")
            return

    def _diff_actions(self, old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
        actions: List[str] = []
        for key in ["dram_bw", "nop_bw", "micro_batch", "tensor_parall"]:
            if old.get(key) != new.get(key):
                actions.append(f"{key}: {old.get(key)}->{new.get(key)}")
        search_space = self._search_space()
        old_size = infer_chip_size(old, search_space)
        new_size = infer_chip_size(new, search_space)
        if old_size != new_size:
            actions.append(f"chip_size: {old_size}->{new_size}")
        if old.get("num_chiplets") != new.get("num_chiplets"):
            actions.append(f"num_chiplets: {old.get('num_chiplets')}->{new.get('num_chiplets')}")
        old_shape = (old.get("chip_y"), old.get("chip_x"))
        new_shape = (new.get("chip_y"), new.get("chip_x"))
        if old_shape != new_shape:
            actions.append(f"chip_shape: {old_shape[0]}x{old_shape[1]}->{new_shape[0]}x{new_shape[1]}")
        old_chiplets = old.get("chiplets", [])
        new_chiplets = new.get("chiplets", [])
        changed_types = 0
        for old_chip, new_chip in zip(old_chiplets, new_chiplets):
            if isinstance(old_chip, dict) and isinstance(new_chip, dict) and old_chip.get("type") != new_chip.get("type"):
                changed_types += 1
        if changed_types:
            actions.append(f"chiplet_type: changed on {changed_types} chiplets")
        return actions

    def _directions_for_bottleneck(self, bottleneck_type: str) -> Dict[str, List[int]]:
        if bottleneck_type == "communication":
            return {"nop_bw": [1], "tensor_parall": [-1, 1], "chiplet_type": [1, -1]}
        if bottleneck_type == "memory":
            return {"dram_bw": [1], "chip_size": [1], "micro_batch": [-1, 1]}
        if bottleneck_type == "compute":
            return {"chip_size": [1], "tensor_parall": [1, -1]}
        if bottleneck_type == "buffer":
            return {"chip_size": [1], "micro_batch": [-1, 1], "chiplet_type": [1, -1]}
        return {}

    def _focus_for_bottleneck(self, bottleneck_type: str) -> List[str]:
        return {
            "compute": ["chip_size", "tensor_parall"],
            "memory": ["dram_bw", "chip_size", "micro_batch"],
            "communication": ["nop_bw", "tensor_parall", "chiplet_type"],
            "buffer": ["chip_size", "micro_batch", "chiplet_type"],
        }.get(bottleneck_type, ["dram_bw", "nop_bw", "chip_size", "chiplet_type"])

    def _search_space(self) -> Dict[str, Any]:
        return self.simulator_schema.get("search_space") or DEFAULT_HARDWARE_SEARCH_SPACE
