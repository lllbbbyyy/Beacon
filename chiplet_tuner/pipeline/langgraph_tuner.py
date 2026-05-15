from __future__ import annotations

import copy
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, TypedDict

from chiplet_tuner.core.io import write_json
from chiplet_tuner.core.schemas import BottleneckState, EvaluationResult, ModelAnalysisResult, SolutionProposal
from chiplet_tuner.pipeline.tuner import MultiAgentTuner
from chiplet_tuner.rag.vector_store import hardware_fingerprint


class LangGraphTuningState(TypedDict, total=False):
    initial_hardware: Dict[str, Any]
    current_hardware: Dict[str, Any]
    iterations: int
    next_iteration: int
    phase: str
    evaluations_by_iteration: Dict[str, Dict[str, Any]]
    applied_changes_by_iteration: Dict[str, Dict[str, Any]]
    evaluated_fingerprints: list[str]
    initial_objective: float
    best: Dict[str, Any]
    iteration_metrics: list[Dict[str, Any]]
    latest_evaluation: Dict[str, Any]
    latest_iteration: int
    latest_change: Dict[str, Any]
    search_state: Dict[str, Any]
    evaluation_bases: Dict[str, Dict[str, Any]]
    forbidden_hardware_fingerprints: list[str]
    iteration_dir: str
    model_result: Dict[str, Any]
    bottleneck_state: Dict[str, Any]
    proposal: Dict[str, Any]
    analysis_evaluation: Dict[str, Any]
    analysis_fingerprint: str
    next_fingerprint: str
    pending_transition: Dict[str, Any]
    current_evaluation: Dict[str, Any]
    summary: Dict[str, Any]
    checkpoint: Dict[str, Any]


class LangGraphTuner(MultiAgentTuner):
    """LangGraph orchestration backend for the iterative tuning pipeline.

    The graph owns the outer control flow and state transitions. Existing agent
    implementations, tools, Compass adapter, trace writer, and RAG store are
    reused so the algorithmic behavior stays aligned with the classic pipeline.
    """

    baseline_stages = 1
    tuning_stages = 7

    def __init__(
        self,
        *args: Any,
        resume: bool = False,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.resume = resume
        self.thread_id = thread_id

    def tune(
        self,
        initial_hardware: Dict[str, Any],
        iterations: int,
    ) -> Dict[str, Any]:
        if self.simulator is None:
            raise ValueError("A SimulatorAdapter is required for iterative tuning.")
        if iterations < 0:
            raise ValueError(f"iterations must be non-negative, got {iterations}")

        graph = self._build_langgraph()
        state: Optional[LangGraphTuningState] = None
        if not self.resume:
            state = {
                "initial_hardware": copy.deepcopy(initial_hardware),
                "current_hardware": copy.deepcopy(initial_hardware),
                "iterations": int(iterations),
            }
        elif not (self.output_root / "langgraph_checkpoints.sqlite").exists():
            raise FileNotFoundError(
                "Cannot resume: no LangGraph SQLite checkpoint found at "
                f"{self.output_root / 'langgraph_checkpoints.sqlite'}"
            )
        with self._checkpointer_context() as checkpointer:
            self._write_langgraph_manifest()
            compiled = graph.compile(checkpointer=checkpointer)
            final_state = compiled.invoke(
                state,
                config={
                    "configurable": {"thread_id": self._checkpoint_thread_id()},
                    "recursion_limit": max(25, int(iterations) * 10 + 10),
                },
            )
        summary = final_state.get("summary")
        if not isinstance(summary, dict):
            raise RuntimeError("LangGraph tuning completed without a final summary.")
        if self.resume:
            summary = copy.deepcopy(summary)
            checkpoint = dict(summary.get("checkpoint", {}))
            checkpoint.update(getattr(self, "_checkpoint_metadata", {}))
            checkpoint["resumed"] = True
            summary["checkpoint"] = checkpoint
            write_json(self.output_root / "tuning_summary.json", summary)
        return summary

    def _checkpoint_thread_id(self) -> str:
        return self.thread_id or str(self.output_root.resolve())

    @contextmanager
    def _checkpointer_context(self) -> Iterator[Any]:
        checkpoint_path = self.output_root / "langgraph_checkpoints.sqlite"
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError:
            from langgraph.checkpoint.memory import MemorySaver

            self._checkpoint_metadata = {
                "type": "memory",
                "path": None,
                "thread_id": self._checkpoint_thread_id(),
                "note": "langgraph-checkpoint-sqlite is not installed; checkpoint is process-local only",
            }
            yield MemorySaver()
            return

        with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            self._checkpoint_metadata = {
                "type": "sqlite",
                "path": str(checkpoint_path),
                "thread_id": self._checkpoint_thread_id(),
            }
            yield saver

    def _build_langgraph(self) -> Any:
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise ImportError(
                "LangGraph pipeline requested, but langgraph is not installed. "
                "Install it in the active environment with: pip install langgraph"
            ) from exc

        graph = StateGraph(LangGraphTuningState)
        graph.add_node("baseline", self._node_baseline)
        graph.add_node("prepare_iteration", self._node_prepare_iteration)
        graph.add_node("model_level_agent", self._node_model_level_agent)
        graph.add_node("layer_level_agent", self._node_layer_level_agent)
        graph.add_node("solution_generation_agent", self._node_solution_generation_agent)
        graph.add_node("write_trace", self._node_write_trace)
        graph.add_node("repair_duplicate", self._node_repair_duplicate)
        graph.add_node("build_transition", self._node_build_transition)
        graph.add_node("evaluate", self._node_evaluate)
        graph.add_node("commit", self._node_commit)
        graph.add_node("finalize", self._node_finalize)

        graph.set_entry_point("baseline")
        graph.add_edge("baseline", "prepare_iteration")
        graph.add_conditional_edges(
            "prepare_iteration",
            self._route_after_prepare,
            {
                "continue": "model_level_agent",
                "finish": "finalize",
            },
        )
        graph.add_edge("model_level_agent", "layer_level_agent")
        graph.add_edge("layer_level_agent", "solution_generation_agent")
        graph.add_edge("solution_generation_agent", "write_trace")
        graph.add_edge("write_trace", "repair_duplicate")
        graph.add_edge("repair_duplicate", "build_transition")
        graph.add_edge("build_transition", "evaluate")
        graph.add_edge("evaluate", "commit")
        graph.add_edge("commit", "prepare_iteration")
        graph.add_edge("finalize", END)
        return graph

    def _node_baseline(self, state: LangGraphTuningState) -> Dict[str, Any]:
        current_hardware = copy.deepcopy(state["current_hardware"])
        baseline_fingerprint = hardware_fingerprint(current_hardware)
        started = time.monotonic()
        try:
            with self.progress.task(
                iteration=0,
                total_iterations=None,
                stage=1,
                total_stages=self.baseline_stages,
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

        initial_objective = self._objective(baseline_evaluation.metrics)
        baseline_metrics = self._metrics_with_objective(baseline_evaluation.metrics)
        baseline_change = self._initial_change_summary()
        best = {
            "score": initial_objective,
            "hardware": copy.deepcopy(current_hardware),
            "metrics": baseline_metrics,
            "iteration": 0,
            "phase": "baseline",
        }
        iteration_metrics = [
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
        ]
        write_json(self.output_root / "best_hardware.json", current_hardware)
        self._write_tuning_metrics_outputs(iteration_metrics)
        updates: Dict[str, Any] = {
            "current_hardware": current_hardware,
            "evaluations_by_iteration": {"0": self._evaluation_record(baseline_evaluation)},
            "applied_changes_by_iteration": {"0": baseline_change},
            "evaluated_fingerprints": [baseline_fingerprint],
            "initial_objective": initial_objective,
            "best": best,
            "iteration_metrics": iteration_metrics,
            "next_iteration": 1,
            "checkpoint": getattr(self, "_checkpoint_metadata", {}),
        }
        self._write_graph_state_marker({**state, **updates}, "baseline")
        return updates

    def _node_prepare_iteration(self, state: LangGraphTuningState) -> Dict[str, Any]:
        iteration = int(state.get("next_iteration", 1))
        iterations = int(state["iterations"])
        if iteration > iterations:
            updates = {"next_iteration": iteration}
            self._write_graph_state_marker({**state, **updates}, "prepare_iteration")
            return updates

        evaluations_by_iteration = self._evaluations_from_records(state["evaluations_by_iteration"])
        applied_changes_by_iteration = state["applied_changes_by_iteration"]
        latest_iteration = max(evaluations_by_iteration)
        latest_evaluation = evaluations_by_iteration[latest_iteration]
        latest_change = applied_changes_by_iteration.get(str(latest_iteration), self._unknown_change_summary())
        forbidden = set(state["evaluated_fingerprints"]) | self.history_store.all_hardware_fingerprints()
        search_state, evaluation_bases = self._build_search_state(
            current_iteration=latest_iteration,
            evaluations_by_iteration=evaluations_by_iteration,
            applied_change=latest_change,
        )
        updates = {
            "phase": "guided",
            "latest_iteration": latest_iteration,
            "latest_evaluation": self._evaluation_record(latest_evaluation),
            "latest_change": latest_change,
            "forbidden_hardware_fingerprints": sorted(forbidden),
            "search_state": search_state,
            "evaluation_bases": self._evaluation_base_records(evaluation_bases),
            "iteration_dir": str(self.output_root / f"iter_{iteration:03d}"),
        }
        self._write_graph_state_marker({**state, **updates}, "prepare_iteration")
        return updates

    def _route_after_prepare(self, state: LangGraphTuningState) -> str:
        return "finish" if int(state.get("next_iteration", 1)) > int(state["iterations"]) else "continue"

    def _node_model_level_agent(self, state: LangGraphTuningState) -> Dict[str, Any]:
        iteration = int(state["next_iteration"])
        latest_evaluation = self._evaluation_from_record(state["latest_evaluation"])
        evaluation_bases = self._evaluation_bases_from_records(state["evaluation_bases"])
        iteration_dir = Path(state["iteration_dir"])
        self._configure_llm_trace(iteration_dir)
        self._validate_required_metrics(latest_evaluation.metrics)
        started = time.monotonic()
        try:
            with self.progress.task(
                iteration=iteration,
                total_iterations=int(state["iterations"]),
                stage=1,
                total_stages=self.tuning_stages,
                component="agent:model_level",
                action="analyze model-level bottlenecks",
            ):
                model_result = self.model_agent.analyze(
                    latest_evaluation,
                    iteration_dir,
                    search_state=state["search_state"],
                    evaluation_bases=evaluation_bases,
                )
        finally:
            self._record_timing(
                iteration=iteration,
                component="agent:model_level",
                action="analyze model-level bottlenecks",
                duration_s=time.monotonic() - started,
            )
        analysis_evaluation = self._evaluation_for_analysis_base(
            model_result,
            evaluation_bases,
            default=latest_evaluation,
        )
        updates = {
            "model_result": self._model_result_record(model_result),
            "analysis_evaluation": self._evaluation_record(analysis_evaluation),
            "analysis_fingerprint": hardware_fingerprint(analysis_evaluation.hardware),
        }
        self._write_graph_state_marker({**state, **updates}, "model_level_agent")
        return updates

    def _node_layer_level_agent(self, state: LangGraphTuningState) -> Dict[str, Any]:
        iteration = int(state["next_iteration"])
        model_result = self._model_result_from_record(state["model_result"])
        analysis_evaluation = self._evaluation_from_record(state["analysis_evaluation"])
        self._configure_llm_trace(Path(state["iteration_dir"]))
        started = time.monotonic()
        try:
            with self.progress.task(
                iteration=iteration,
                total_iterations=int(state["iterations"]),
                stage=2,
                total_stages=self.tuning_stages,
                component="agent:layer_level",
                action="diagnose candidate layers",
            ):
                bottleneck_state = self.layer_agent.analyze(
                    model_result,
                    analysis_evaluation,
                    Path(state["iteration_dir"]),
                )
        finally:
            self._record_timing(
                iteration=iteration,
                component="agent:layer_level",
                action="diagnose candidate layers",
                duration_s=time.monotonic() - started,
            )
        updates = {"bottleneck_state": self._bottleneck_state_record(bottleneck_state)}
        self._write_graph_state_marker({**state, **updates}, "layer_level_agent")
        return updates

    def _node_solution_generation_agent(self, state: LangGraphTuningState) -> Dict[str, Any]:
        iteration = int(state["next_iteration"])
        bottleneck_state = self._bottleneck_state_from_record(state["bottleneck_state"])
        analysis_evaluation = self._evaluation_from_record(state["analysis_evaluation"])
        evaluation_bases = self._evaluation_bases_from_records(state["evaluation_bases"])
        self._configure_llm_trace(Path(state["iteration_dir"]))
        started = time.monotonic()
        try:
            with self.progress.task(
                iteration=iteration,
                total_iterations=int(state["iterations"]),
                stage=3,
                total_stages=self.tuning_stages,
                component="agent:solution_generation",
                action="retrieve history and propose hardware update",
            ):
                proposal = self.solution_agent.propose(
                    bottleneck_state,
                    analysis_evaluation.hardware,
                    output_dir=Path(state["iteration_dir"]),
                    evaluation=analysis_evaluation,
                    forbidden_hardware_fingerprints=set(state["forbidden_hardware_fingerprints"]),
                    search_state=state["search_state"],
                    evaluation_bases=evaluation_bases,
                )
        finally:
            self._record_timing(
                iteration=iteration,
                component="agent:solution_generation",
                action="retrieve history and propose hardware update",
                duration_s=time.monotonic() - started,
            )
        updates = {
            "proposal": self._proposal_record(proposal),
            "next_fingerprint": hardware_fingerprint(proposal.updated_hardware),
        }
        self._write_graph_state_marker({**state, **updates}, "solution_generation_agent")
        return updates

    def _node_write_trace(self, state: LangGraphTuningState) -> Dict[str, Any]:
        iteration = int(state["next_iteration"])
        iteration_dir = Path(state["iteration_dir"])
        trace_path = iteration_dir / "analysis" / "agent_trace.json"
        analysis_evaluation = self._evaluation_from_record(state["analysis_evaluation"])
        model_result = self._model_result_from_record(state["model_result"])
        bottleneck_state = self._bottleneck_state_from_record(state["bottleneck_state"])
        proposal = self._proposal_from_record(state["proposal"])
        started = time.monotonic()
        try:
            with self.progress.task(
                iteration=iteration,
                total_iterations=int(state["iterations"]),
                stage=4,
                total_stages=self.tuning_stages,
                component="trace",
                action="write agent trace and LLM trace index",
            ):
                artifacts = self._collect_trace_artifacts(
                    model_result,
                    bottleneck_state,
                    proposal,
                    trace_path,
                    iteration_dir,
                )
                analysis_payload = self._build_agent_trace_payload(
                    iteration=iteration,
                    evaluation=analysis_evaluation,
                    model_result=model_result,
                    state=bottleneck_state,
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
        updates = {"model_result": self._model_result_record(model_result)}
        self._write_graph_state_marker({**state, **updates}, "write_trace")
        return updates

    def _node_repair_duplicate(self, state: LangGraphTuningState) -> Dict[str, Any]:
        proposal = self._proposal_from_record(state["proposal"])
        next_fingerprint = state["next_fingerprint"]
        iteration = int(state["next_iteration"])
        bottleneck_state = self._bottleneck_state_from_record(state["bottleneck_state"])
        is_duplicate = next_fingerprint in state["evaluated_fingerprints"]
        action = "repair duplicate hardware proposal" if is_duplicate else "validate hardware proposal uniqueness"
        repaired: Optional[Dict[str, Any]] = None
        repair_actions: list[str] = []
        started = time.monotonic()
        try:
            with self.progress.task(
                iteration=iteration,
                total_iterations=int(state["iterations"]),
                stage=5,
                total_stages=self.tuning_stages,
                component="pipeline",
                action=action,
                detail=f"hardware_fingerprint={next_fingerprint}, duplicate={is_duplicate}",
            ):
                if is_duplicate:
                    repaired, repair_actions = self.solution_agent.make_exploration_move(
                        proposal.updated_hardware,
                        bottleneck_state,
                        forbidden_hardware_fingerprints=set(state["forbidden_hardware_fingerprints"]),
                    )
        finally:
            self._record_timing(
                iteration=iteration,
                component="pipeline",
                action=action,
                duration_s=time.monotonic() - started,
                metadata={"hardware_fingerprint": next_fingerprint, "duplicate": is_duplicate},
            )
        repaired_fingerprint = hardware_fingerprint(repaired) if repaired is not None else next_fingerprint
        if repaired is not None and repaired_fingerprint != next_fingerprint:
            proposal.updated_hardware = repaired
            proposal.actions.extend(action for action in repair_actions if action not in proposal.actions)
            proposal.llm_notes.setdefault("validation", []).append(
                "pipeline repaired duplicate next hardware after proposal"
            )
            next_fingerprint = repaired_fingerprint
        updates = {"proposal": self._proposal_record(proposal), "next_fingerprint": next_fingerprint}
        self._write_graph_state_marker({**state, **updates}, "repair_duplicate")
        return updates

    def _node_build_transition(self, state: LangGraphTuningState) -> Dict[str, Any]:
        iteration = int(state["next_iteration"])
        analysis_evaluation = self._evaluation_from_record(state["analysis_evaluation"])
        model_result = self._model_result_from_record(state["model_result"])
        bottleneck_state = self._bottleneck_state_from_record(state["bottleneck_state"])
        proposal = self._proposal_from_record(state["proposal"])
        pending_transition = self._build_pending_transition(
            iteration=iteration,
            phase=str(state["phase"]),
            evaluation=analysis_evaluation,
            model_result=model_result,
            state=bottleneck_state,
            proposal=proposal,
            current_hardware=analysis_evaluation.hardware,
            current_fingerprint=state["analysis_fingerprint"],
            next_fingerprint=state["next_fingerprint"],
        )
        write_json(self.output_root / "pending_transition.json", pending_transition)
        write_json(self.output_root / "latest_proposal.json", asdict(proposal))
        updates = {"pending_transition": pending_transition}
        self._write_graph_state_marker({**state, **updates}, "build_transition")
        return updates

    def _node_evaluate(self, state: LangGraphTuningState) -> Dict[str, Any]:
        iteration = int(state["next_iteration"])
        proposal = self._proposal_from_record(state["proposal"])
        started = time.monotonic()
        try:
            with self.progress.task(
                iteration=iteration,
                total_iterations=int(state["iterations"]),
                stage=6,
                total_stages=self.tuning_stages,
                component="evaluator",
                action="evaluate proposed hardware",
                detail=f"hardware_fingerprint={state['next_fingerprint']}",
            ):
                evaluation = self.simulator.evaluate(proposal.updated_hardware, iteration)
        finally:
            self._record_timing(
                iteration=iteration,
                component="evaluator",
                action="evaluate proposed hardware",
                duration_s=time.monotonic() - started,
                metadata={"hardware_fingerprint": state["next_fingerprint"]},
            )
        updates = {"current_evaluation": self._evaluation_record(evaluation)}
        self._write_graph_state_marker({**state, **updates}, "evaluate")
        return updates

    def _node_commit(self, state: LangGraphTuningState) -> Dict[str, Any]:
        iteration = int(state["next_iteration"])
        evaluation = self._evaluation_from_record(state["current_evaluation"])
        proposal = self._proposal_from_record(state["proposal"])
        bottleneck_state = self._bottleneck_state_from_record(state["bottleneck_state"])
        pending_transition = state["pending_transition"]
        started = time.monotonic()
        try:
            with self.progress.task(
                iteration=iteration,
                total_iterations=int(state["iterations"]),
                stage=7,
                total_stages=self.tuning_stages,
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

        evaluations_by_iteration = dict(state["evaluations_by_iteration"])
        evaluations_by_iteration[str(iteration)] = self._evaluation_record(evaluation)
        evaluated_fingerprints = set(state["evaluated_fingerprints"])
        evaluated_fingerprints.add(state["next_fingerprint"])
        applied_changes_by_iteration = dict(state["applied_changes_by_iteration"])
        applied_change = self._change_summary_from_transition(pending_transition)
        applied_changes_by_iteration[str(iteration)] = applied_change

        score = self._objective(evaluation.metrics)
        best = dict(state["best"])
        previous_best_score = float(best["score"])
        is_new_best = score < previous_best_score
        metrics = self._metrics_with_objective(evaluation.metrics)
        iteration_metrics = list(state["iteration_metrics"])
        iteration_metrics.append(
            {
                "iteration": iteration,
                "phase": state["phase"],
                "metrics": metrics,
                "objective_change_vs_initial": self._change_ratio(state["initial_objective"], score),
                "objective_change_vs_previous_best": self._change_ratio(previous_best_score, score),
                "is_new_best": is_new_best,
                "applied_change": applied_change,
                "hardware_fingerprint": state["next_fingerprint"],
            }
        )
        current_hardware = proposal.updated_hardware
        if is_new_best:
            best = {
                "score": score,
                "hardware": copy.deepcopy(current_hardware),
                "metrics": metrics,
                "iteration": iteration,
                "phase": state["phase"],
            }
            write_json(self.output_root / "best_hardware.json", current_hardware)
        write_json(
            self.output_root / "pending_transition.json",
            {
                "status": "evaluated",
                "source_iteration": pending_transition.get("source_iteration"),
                "after_iteration": iteration,
                "hardware_fingerprint": state["next_fingerprint"],
                "before_metrics": pending_transition.get("before_metrics"),
                "after_metrics": metrics,
                "applied_change": applied_change,
            },
        )
        self._write_tuning_metrics_outputs(iteration_metrics)
        self.progress.info(
            iteration=iteration,
            total_iterations=int(state["iterations"]),
            component="iteration",
            message=(
                f"DONE phase={state['phase']} "
                f"bottleneck={bottleneck_state.primary_impact}/{bottleneck_state.dominant_root_cause} "
                f"actions={', '.join(proposal.actions)}"
            ),
        )
        updates = {
            "current_hardware": current_hardware,
            "evaluations_by_iteration": evaluations_by_iteration,
            "evaluated_fingerprints": sorted(evaluated_fingerprints),
            "applied_changes_by_iteration": applied_changes_by_iteration,
            "iteration_metrics": iteration_metrics,
            "best": best,
            "next_iteration": iteration + 1,
        }
        self._write_graph_state_marker({**state, **updates}, "commit")
        return updates

    def _node_finalize(self, state: LangGraphTuningState) -> Dict[str, Any]:
        metrics_table_path = self._write_tuning_metrics_outputs(state["iteration_metrics"])
        timing_summary = self._write_timing_outputs()
        final = {
            "best": state.get("best", {}),
            "next_hardware": state.get("current_hardware", {}),
            "evaluated_designs": len(state.get("evaluated_fingerprints", [])),
            "iterations": int(state["iterations"]),
            "tuning_iterations": int(state["iterations"]),
            "proposal_iterations": int(state["iterations"]),
            "evaluated_runs": len(state.get("iteration_metrics", [])),
            "history_records": len(self.history_store),
            "llm_retry_summary": self._llm_retry_summary(),
            "metrics_table": str(metrics_table_path),
            "iteration_metrics": state.get("iteration_metrics", []),
            "timing_summary": timing_summary,
            "timing_summary_json": str(self.output_root / "timing_summary.json"),
            "timing_summary_md": str(self.output_root / "timing_summary.md"),
            "pipeline_backend": "langgraph",
            "checkpoint": {
                **getattr(self, "_checkpoint_metadata", state.get("checkpoint", {})),
                "resumed": self.resume,
            },
        }
        write_json(self.output_root / "tuning_summary.json", final)
        updates = {"summary": final}
        self._write_graph_state_marker({**state, **updates}, "finalize")
        return updates

    def _evaluation_record(self, evaluation: EvaluationResult) -> Dict[str, Any]:
        return {
            "run_dir": str(evaluation.run_dir),
            "hardware": copy.deepcopy(evaluation.hardware),
            "metrics": copy.deepcopy(evaluation.metrics),
            "detail_files": {key: str(path) for key, path in evaluation.detail_files.items()},
            "raw_files": {key: str(path) for key, path in evaluation.raw_files.items()},
        }

    def _evaluation_from_record(self, record: Dict[str, Any]) -> EvaluationResult:
        if not isinstance(record, dict):
            raise TypeError(f"Expected evaluation record dict, got {type(record).__name__}")
        return EvaluationResult(
            run_dir=Path(str(record["run_dir"])),
            hardware=copy.deepcopy(record["hardware"]),
            metrics=copy.deepcopy(record["metrics"]),
            detail_files={key: Path(str(path)) for key, path in record.get("detail_files", {}).items()},
            raw_files={key: Path(str(path)) for key, path in record.get("raw_files", {}).items()},
        )

    def _evaluations_from_records(self, records: Dict[str, Dict[str, Any]]) -> Dict[int, EvaluationResult]:
        return {int(iteration): self._evaluation_from_record(record) for iteration, record in records.items()}

    def _evaluation_base_records(
        self,
        bases: Dict[str, EvaluationResult],
    ) -> Dict[str, Dict[str, Any]]:
        return {key: self._evaluation_record(evaluation) for key, evaluation in bases.items()}

    def _evaluation_bases_from_records(
        self,
        records: Dict[str, Dict[str, Any]],
    ) -> Dict[str, EvaluationResult]:
        return {key: self._evaluation_from_record(record) for key, record in records.items()}

    def _model_result_record(self, result: ModelAnalysisResult) -> Dict[str, Any]:
        return asdict(result)

    def _model_result_from_record(self, record: Dict[str, Any]) -> ModelAnalysisResult:
        return ModelAnalysisResult(
            metrics=copy.deepcopy(record.get("metrics", {})),
            candidate_layers=copy.deepcopy(record.get("candidate_layers", [])),
            generated_files=copy.deepcopy(record.get("generated_files", {})),
            summary=str(record.get("summary", "")),
            global_findings=copy.deepcopy(record.get("global_findings", [])),
            selected_views=copy.deepcopy(record.get("selected_views", [])),
            analysis_base=copy.deepcopy(record.get("analysis_base", {})),
            llm_notes=copy.deepcopy(record.get("llm_notes", {})),
        )

    def _bottleneck_state_record(self, state: BottleneckState) -> Dict[str, Any]:
        return asdict(state)

    def _bottleneck_state_from_record(self, record: Dict[str, Any]) -> BottleneckState:
        return BottleneckState(
            primary_impact=str(record.get("primary_impact", "unknown")),
            dominant_root_cause=str(record.get("dominant_root_cause", "unknown")),
            layer_diagnoses=copy.deepcopy(record.get("layer_diagnoses", [])),
            retrieval_description=str(record.get("retrieval_description", "")),
            root_cause_summary=copy.deepcopy(record.get("root_cause_summary", {})),
            recommended_focus=copy.deepcopy(record.get("recommended_focus", [])),
            llm_notes=copy.deepcopy(record.get("llm_notes", {})),
        )

    def _proposal_record(self, proposal: SolutionProposal) -> Dict[str, Any]:
        return asdict(proposal)

    def _proposal_from_record(self, record: Dict[str, Any]) -> SolutionProposal:
        return SolutionProposal(
            strategy=str(record.get("strategy", "")),
            updated_hardware=copy.deepcopy(record.get("updated_hardware", {})),
            actions=copy.deepcopy(record.get("actions", [])),
            retrieved_cases=copy.deepcopy(record.get("retrieved_cases", [])),
            rationale=str(record.get("rationale", "")),
            llm_notes=copy.deepcopy(record.get("llm_notes", {})),
        )

    def _write_graph_state_marker(self, state: Dict[str, Any], node: str) -> None:
        payload = {
            "backend": "langgraph",
            "last_node": node,
            "iterations": state.get("iterations"),
            "next_iteration": state.get("next_iteration"),
            "completed_evaluations": len(state.get("evaluations_by_iteration", {})),
            "evaluated_designs": len(state.get("evaluated_fingerprints", [])),
            "checkpoint": {
                **getattr(self, "_checkpoint_metadata", state.get("checkpoint", {})),
                "resumed": self.resume,
            },
            "best": self._compact_best_state(state.get("best")),
            "latest_metrics_row": (
                state.get("iteration_metrics", [])[-1]
                if isinstance(state.get("iteration_metrics"), list) and state.get("iteration_metrics")
                else None
            ),
        }
        write_json(self.output_root / "langgraph_state.json", payload)

    def _write_langgraph_manifest(self) -> None:
        manifest = {
            "backend": "langgraph",
            "state_policy": (
                "Graph state stores JSON-serializable records only; dataclass objects are "
                "reconstructed inside node functions and are not checkpoint payloads."
            ),
            "checkpoint": getattr(self, "_checkpoint_metadata", {}),
            "outer_graph": {
                "nodes": [
                    "baseline",
                    "prepare_iteration",
                    "model_level_agent",
                    "layer_level_agent",
                    "solution_generation_agent",
                    "write_trace",
                    "repair_duplicate",
                    "build_transition",
                    "evaluate",
                    "commit",
                    "finalize",
                ],
                "edges": [
                    ["baseline", "prepare_iteration"],
                    ["prepare_iteration", "model_level_agent|finalize"],
                    ["model_level_agent", "layer_level_agent"],
                    ["layer_level_agent", "solution_generation_agent"],
                    ["solution_generation_agent", "write_trace"],
                    ["write_trace", "repair_duplicate"],
                    ["repair_duplicate", "build_transition"],
                    ["build_transition", "evaluate"],
                    ["evaluate", "commit"],
                    ["commit", "prepare_iteration"],
                    ["finalize", "END"],
                ],
            },
            "agent_react_subgraph": {
                "shared_by_agents": ["model_level", "layer_level", "solution_generation"],
                "nodes": ["llm_decide", "execute_tool"],
                "conditional_edges": [
                    ["llm_decide", "execute_tool|END"],
                    ["execute_tool", "llm_decide|END"],
                ],
                "tool_policy": "At most one tool call per ReAct step; duplicate tool calls end tool use.",
            },
        }
        write_json(self.output_root / "langgraph_manifest.json", manifest)

    def _compact_best_state(self, best: Any) -> Dict[str, Any]:
        if not isinstance(best, dict):
            return {}
        return {
            "iteration": best.get("iteration"),
            "phase": best.get("phase"),
            "score": best.get("score"),
            "metrics": best.get("metrics"),
        }
