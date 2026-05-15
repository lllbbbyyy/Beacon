from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, TextIO, Tuple


class ProgressReporter:
    """Small dependency-free progress reporter for long tuning runs."""

    def __init__(
        self,
        enabled: bool = True,
        stream: Optional[TextIO] = None,
        refresh_interval_s: float = 30.0,
    ) -> None:
        self.enabled = enabled
        self.stream = stream or sys.stdout
        self.refresh_interval_s = refresh_interval_s
        self._run_start = time.monotonic()
        self._duration_history: Dict[Tuple[str, str], List[float]] = {}
        self._stage_keys: Dict[Tuple[int, int], Tuple[str, str]] = {}

    @contextmanager
    def task(
        self,
        *,
        iteration: Optional[int],
        total_iterations: Optional[int],
        stage: int,
        total_stages: int,
        component: str,
        action: str,
        detail: str = "",
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        start = time.monotonic()
        task_key = self._task_key(component, action)
        self._remember_stage_key(total_stages, stage, task_key)
        self._emit(
            status="START",
            iteration=iteration,
            total_iterations=total_iterations,
            stage=stage,
            total_stages=total_stages,
            component=component,
            action=action,
            detail=detail,
            elapsed_s=0.0,
            task_key=task_key,
        )
        try:
            yield
        except Exception:
            self._emit(
                status="FAILED",
                iteration=iteration,
                total_iterations=total_iterations,
                stage=stage,
                total_stages=total_stages,
                component=component,
                action=action,
                detail=detail,
                elapsed_s=time.monotonic() - start,
                task_key=task_key,
            )
            raise
        else:
            elapsed_s = time.monotonic() - start
            self._record_duration(task_key, elapsed_s)
            self._emit(
                status="DONE",
                iteration=iteration,
                total_iterations=total_iterations,
                stage=stage,
                total_stages=total_stages,
                component=component,
                action=action,
                detail=detail,
                elapsed_s=elapsed_s,
                task_key=task_key,
            )

    def skip(
        self,
        *,
        iteration: Optional[int],
        total_iterations: Optional[int],
        stage: int,
        total_stages: int,
        component: str,
        action: str,
        reason: str = "",
    ) -> None:
        if not self.enabled:
            return
        task_key = self._task_key(component, action)
        self._remember_stage_key(total_stages, stage, task_key)
        self._emit(
            status="SKIP",
            iteration=iteration,
            total_iterations=total_iterations,
            stage=stage,
            total_stages=total_stages,
            component=component,
            action=action,
            detail=reason,
            elapsed_s=0.0,
            task_key=task_key,
        )

    def info(
        self,
        *,
        iteration: Optional[int],
        total_iterations: Optional[int],
        component: str,
        message: str,
    ) -> None:
        if not self.enabled:
            return
        iter_label = self._iteration_label(iteration, total_iterations)
        self._write(
            f"[{iter_label}][{self._component_label(component)}][INFO]"
            f"[elapsed={self._fmt_duration(time.monotonic() - self._run_start)}]"
            f"[eta=unknown] {message}"
        )

    def _emit(
        self,
        *,
        status: str,
        iteration: Optional[int],
        total_iterations: Optional[int],
        stage: int,
        total_stages: int,
        component: str,
        action: str,
        detail: str,
        elapsed_s: float,
        task_key: Tuple[str, str],
    ) -> None:
        eta_remaining_s = self._estimate_remaining_s(
            status=status,
            iteration=iteration,
            total_iterations=total_iterations,
            stage=stage,
            total_stages=total_stages,
            task_key=task_key,
            elapsed_s=elapsed_s,
        )
        if eta_remaining_s is None:
            eta_remaining = "unknown"
        else:
            eta_remaining = self._fmt_duration(eta_remaining_s)
        parts = [
            f"[{self._iteration_label(iteration, total_iterations)}]",
            f"[step={stage}/{total_stages}:{self._component_label(component)}]",
            f"[{status}]",
            f"[elapsed={self._fmt_duration(time.monotonic() - self._run_start)}]",
            f"[eta={eta_remaining}]",
        ]
        if status in {"DONE", "FAILED"}:
            parts.append(f"[step_time={self._fmt_duration(elapsed_s)}]")
        suffix = f" reason={detail}" if status == "SKIP" and detail else (f" detail={detail}" if detail else "")
        self._write(
            "".join(parts) + f" {action}{suffix}"
        )

    def _task_key(self, component: str, action: str) -> Tuple[str, str]:
        return (component, action)

    def _remember_stage_key(self, total_stages: int, stage: int, task_key: Tuple[str, str]) -> None:
        self._stage_keys.setdefault((total_stages, stage), task_key)

    def _record_duration(self, task_key: Tuple[str, str], elapsed_s: float) -> None:
        self._duration_history.setdefault(task_key, []).append(elapsed_s)

    def _estimate_remaining_s(
        self,
        *,
        status: str,
        iteration: Optional[int],
        total_iterations: Optional[int],
        stage: int,
        total_stages: int,
        task_key: Tuple[str, str],
        elapsed_s: float,
    ) -> Optional[float]:
        stage_keys = dict(self._stage_keys)
        duration_history = {key: list(value) for key, value in self._duration_history.items()}
        estimate = 0.0

        if status != "DONE":
            average = self._average_duration(task_key, duration_history)
            if average is None:
                return None
            estimate += max(average - elapsed_s, 0.0)

        for future_stage in range(stage + 1, total_stages + 1):
            average = self._average_duration(stage_keys.get((total_stages, future_stage)), duration_history)
            if average is None:
                return None
            estimate += average

        if iteration is not None and total_iterations is not None:
            if iteration <= 0:
                remaining_full_iterations = total_iterations
            else:
                remaining_full_iterations = max(total_iterations - iteration, 0)
            if remaining_full_iterations:
                full_iteration_estimate = 0.0
                for future_stage in range(1, total_stages + 1):
                    average = self._average_duration(stage_keys.get((total_stages, future_stage)), duration_history)
                    if average is None:
                        return None
                    full_iteration_estimate += average
                estimate += remaining_full_iterations * full_iteration_estimate

        return estimate

    def _average_duration(
        self,
        task_key: Optional[Tuple[str, str]],
        duration_history: Dict[Tuple[str, str], List[float]],
    ) -> Optional[float]:
        if task_key is None:
            return None
        samples = duration_history.get(task_key, [])
        if not samples:
            return None
        return sum(samples) / len(samples)

    def _write(self, line: str) -> None:
        print(line, file=self.stream, flush=True)

    def _iteration_label(self, iteration: Optional[int], total_iterations: Optional[int]) -> str:
        if iteration is None:
            return "iter=n/a"
        if iteration <= 0:
            return "iter=init"
        if total_iterations is None:
            return f"iter={iteration}"
        return f"iter={iteration}/{total_iterations}"

    def _component_label(self, component: str) -> str:
        labels = {
            "agent:model_level": "model_agent",
            "agent:layer_level": "layer_agent",
            "agent:solution_generation": "solution_agent",
        }
        return labels.get(component, component)

    def _fmt_duration(self, seconds: float) -> str:
        total = int(max(seconds, 0.0))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
