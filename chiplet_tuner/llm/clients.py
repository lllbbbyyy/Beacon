from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Optional, Sequence

from chiplet_tuner.core.schemas import LLMConfig
from chiplet_tuner.llm.tracing import LLMTraceRecorder


class LLMClient:
    trace_enabled: bool = False
    trace_recorder: Optional[LLMTraceRecorder] = None
    last_reasoning_content: Optional[str] = None
    retry_attempts: int = 0
    retry_temperature: float = 0.1
    temperature_override: Optional[float] = None

    def complete_json_messages(self, messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError

    def complete_json(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.complete_json_messages(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ]
        )

    def assistant_message_from_response(self, response: Dict[str, Any], content: str) -> Dict[str, Any]:
        return {"role": "assistant", "content": content}

    def record_retry_event(self, event: Dict[str, Any]) -> None:
        events = getattr(self, "retry_events", None)
        if events is None:
            events = []
            self.retry_events = events
        events.append(event)

    def retry_summary(self) -> Dict[str, Any]:
        events = list(getattr(self, "retry_events", []))
        by_task: Dict[str, int] = {}
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
            by_task[task] = by_task.get(task, 0) + 1
            error_type = str(event.get("error_type", "unknown"))
            by_error_type[error_type] = by_error_type.get(error_type, 0) + 1
        return {
            "event_count": len(events),
            "retry_attempt_count": retry_attempt_count,
            "exhausted_count": exhausted_count,
            "by_task": by_task,
            "by_error_type": by_error_type,
            "events": events,
        }


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract a JSON object from plain text or fenced markdown."""

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _sdk_base_url(base_url: str) -> str:
    """OpenAI SDK expects the API root, not the final /chat/completions path."""

    suffix = "/chat/completions"
    if base_url.endswith(suffix):
        return base_url[: -len(suffix)].rstrip("/")
    return base_url


def _dump_openai_sdk_response(response: Any) -> Dict[str, Any]:
    if hasattr(response, "model_dump"):
        dumped = response.model_dump(mode="json")
    elif hasattr(response, "to_dict_recursive"):
        dumped = response.to_dict_recursive()
    elif hasattr(response, "to_dict"):
        dumped = response.to_dict()
    else:
        dumped = json.loads(json.dumps(response, default=str))
    if not isinstance(dumped, dict):
        raise RuntimeError(f"OpenAI SDK response dump must be an object, got {type(dumped).__name__}")
    return dumped


def _response_reasoning_content(response: Any) -> Optional[str]:
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return None
    value = getattr(message, "reasoning_content", None)
    if isinstance(value, str) and value:
        return value
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        value = extra.get("reasoning_content")
        if isinstance(value, str) and value:
            return value
    return None


def _openai_sdk_error_payload(exc: BaseException) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    for attr in ("status_code", "code", "param", "type"):
        value = getattr(exc, attr, None)
        if value not in (None, ""):
            payload[attr] = value
    body = getattr(exc, "body", None)
    if body not in (None, ""):
        payload["body"] = body
    response = getattr(exc, "response", None)
    if response is not None:
        payload["response"] = {
            "status_code": getattr(response, "status_code", None),
            "text": str(getattr(response, "text", ""))[:2000],
        }
    return payload


def _is_openai_sdk_retryable_error(exc: BaseException) -> bool:
    error_type = type(exc).__name__
    if error_type in {"APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError"}:
        return True
    status_code = getattr(exc, "status_code", None)
    try:
        status = int(status_code)
    except (TypeError, ValueError):
        return False
    return status in {408, 409, 429} or status >= 500


class OpenAICompatibleClient(LLMClient):
    """Chat-completions client backed by the OpenAI Python SDK."""

    def __init__(self, config: LLMConfig) -> None:
        if not config.api_key:
            raise ValueError("LLM api_key is required.")
        if not config.model:
            raise ValueError("LLM model is required.")
        if not config.base_url:
            raise ValueError("LLM base_url is required.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The openai package is required for real LLM calls. "
                "Install it in the active environment with: pip install openai"
            ) from exc
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.sdk_base_url = _sdk_base_url(self.base_url)
        self.chat_completions_url = (
            self.base_url if self.base_url.endswith("/chat/completions") else f"{self.base_url}/chat/completions"
        )
        self.sdk_client = OpenAI(
            api_key=config.api_key,
            base_url=self.sdk_base_url,
            timeout=float(config.timeout),
        )
        self.trace_enabled = False
        self.trace_recorder: Optional[LLMTraceRecorder] = None
        self.last_reasoning_content: Optional[str] = None
        self.retry_attempts = max(0, int(config.retry_attempts))
        self.retry_temperature = float(config.retry_temperature)
        self.temperature_override: Optional[float] = None
        self.retry_events: List[Dict[str, Any]] = []

    def complete_json_messages(self, messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        temperature = self.temperature_override if self.temperature_override is not None else self.config.temperature
        body = {
            "model": self.config.model,
            "messages": [self._normalize_message(message) for message in messages],
            "temperature": temperature,
        }
        if self.config.max_tokens is not None:
            body["max_tokens"] = self.config.max_tokens
        trace_call = self._trace_begin(body)
        try:
            response = self.sdk_client.chat.completions.create(**body)
        except Exception as exc:
            error_payload = _openai_sdk_error_payload(exc)
            retryable = _is_openai_sdk_retryable_error(exc)
            message = (
                f"LLM request failed via OpenAI SDK "
                f"(type={error_payload['type']}, retryable={retryable}): {error_payload['message']}"
            )
            self._trace_end(
                trace_call,
                {
                    "sdk_error": error_payload,
                    "retryable": retryable,
                },
                error=message,
            )
            raise RuntimeError(message) from exc
        payload = _dump_openai_sdk_response(response)
        try:
            message = payload["choices"][0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            message = f"LLM response did not contain choices[0].message.content: {exc}"
            self._trace_end(trace_call, {"response": payload}, error=message)
            raise RuntimeError(message) from exc
        if not isinstance(content, str):
            message = f"LLM response content must be a string, got {type(content).__name__}"
            self._trace_end(trace_call, {"response": payload}, error=message)
            raise RuntimeError(message)
        reasoning_content = message.get("reasoning_content")
        if not isinstance(reasoning_content, str):
            reasoning_content = _response_reasoning_content(response)
        self.last_reasoning_content = reasoning_content if isinstance(reasoning_content, str) else None
        try:
            parsed = extract_json_object(content)
        except Exception as exc:
            self._trace_end(
                trace_call,
                {
                    "response": payload,
                    "content": content,
                    "parse_error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                },
                response_content=content,
                error=exc,
            )
            raise
        self._trace_end(
            trace_call,
            {
                "response": payload,
                "content": content,
            },
            response_content=content,
        )
        return parsed

    def complete_json(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        return super().complete_json(system_prompt, user_payload)

    def assistant_message_from_response(self, response: Dict[str, Any], content: str) -> Dict[str, Any]:
        message: Dict[str, Any] = {"role": "assistant", "content": content}
        if self.config.return_reasoning and self.last_reasoning_content:
            message["reasoning_content"] = self.last_reasoning_content
        return message

    def _normalize_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported chat message role: {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"Chat message content must be a string, got {type(content).__name__}")
        normalized: Dict[str, Any] = {"role": role, "content": content}
        reasoning_content = message.get("reasoning_content")
        if self.config.return_reasoning and isinstance(reasoning_content, str) and reasoning_content:
            normalized["reasoning_content"] = reasoning_content
        return normalized

    def _trace_begin(self, request_body: Dict[str, Any]) -> Optional[int]:
        if self.trace_recorder is None:
            return None
        return self.trace_recorder.begin_call(
            {
                "url": self.chat_completions_url,
                "request_body": request_body,
            }
        )

    def _trace_end(
        self,
        call_index: Optional[int],
        raw_output: Dict[str, Any],
        response_content: Optional[str] = None,
        error: Optional[BaseException | str] = None,
    ) -> None:
        if call_index is None or self.trace_recorder is None:
            return
        self.trace_recorder.end_call(
            call_index,
            raw_output=raw_output,
            response_content=response_content,
            error=error,
        )


class MockLLMClient(LLMClient):
    """Deterministic local client for smoke tests and offline demos."""

    def complete_json_messages(self, messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        user_payload = _mock_payload_from_messages(messages)
        return self.complete_json("", user_payload)

    def complete_json(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        task = user_payload.get("task")
        if task == "react_tool_use":
            return _mock_react_tool_use(user_payload)
        if task == "model_tool_selection":
            return {
                "analysis_intent": "Collect global model-level evidence from metrics, timeline, rank views, operators, cost, and hardware.",
                "tools": [
                    {"name": "summarize_metrics", "arguments": {}},
                    {"name": "build_execution_timeline", "arguments": {}},
                    {
                        "name": "summarize_layer_rank_views",
                        "arguments": {"top_layers": 6},
                    },
                    {"name": "summarize_operator_groups", "arguments": {}},
                    {"name": "summarize_monetary_cost", "arguments": {}},
                    {"name": "summarize_hardware_config", "arguments": {}},
                ],
            }
        if task == "model_level_analysis":
            metrics = user_payload.get("evaluation", {}).get("metrics", {})
            tool_payload = user_payload["tool_results"]["summarize_layer_rank_views"]["payload"]
            rank_views = tool_payload.get("layer_rank_views", {})
            bottlenecks = rank_views.get("latency_sum", [])
            top_names = [item.get("layer_name") for item in bottlenecks[:3]]
            candidates = [_candidate_from_rank_item(item, concern_type="latency") for item in bottlenecks]
            return {
                "summary": (
                    f"model latency={metrics.get('latency', 0.0):.4g}, "
                    f"energy={metrics.get('energy', 0.0):.4g}, mc={metrics.get('mc', 0.0):.4g}; "
                    "mock selected latency_sum view candidate layers: "
                    + ", ".join(str(name) for name in top_names)
                ),
                "candidate_layers": candidates,
                "notes": {
                    "method": "mock_llm_selected_rank_view",
                    "selected_rank_views": ["latency_sum"],
                    "selected_bottleneck_objective": "latency",
                    "critical_layers": top_names,
                    "global_findings": ["FFN tiling layers dominate the latency_sum rank view."],
                },
            }
        if task == "layer_tool_selection":
            return {
                "analysis_intent": "Cross-check layer diagnosis with operator-group and hardware context.",
                "tools": [
                    {"name": "summarize_operator_groups", "arguments": {}},
                    {"name": "summarize_hardware_config", "arguments": {}},
                ],
            }
        if task == "layer_level_analysis":
            return _mock_layer_level(user_payload)
        if task == "solution_tool_selection":
            return {
                "analysis_intent": "Inspect hardware search-space position and materialize a legal candidate before proposing updates.",
                "tools": [
                    {"name": "summarize_hardware_config", "arguments": {}},
                ],
            }
        if task == "solution_generation":
            return _mock_solution(user_payload)
        raise ValueError(f"MockLLMClient does not support task: {task}")


def _mock_payload_from_messages(messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    parsed_user_messages: List[Dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        final_task = _task_from_final_markdown(message.get("content", ""))
        if final_task:
            parsed_user_messages.append({"task": final_task})
            continue
        try:
            payload = json.loads(message.get("content", ""))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            parsed_user_messages.append(payload)

    for payload in reversed(parsed_user_messages):
        task = payload.get("task")
        if task and task != "tool_observation":
            if task == "react_tool_use":
                return _mock_react_payload_from_messages(payload, parsed_user_messages)
            return _mock_final_payload_from_messages(payload, parsed_user_messages)
    raise ValueError("MockLLMClient messages must include a user JSON payload with a task field.")


def _task_from_final_markdown(content: Any) -> str:
    if not isinstance(content, str):
        return ""
    match = re.search(r"Final (?:decision task|synthesis step):\s*`([^`]+)`", content)
    return match.group(1) if match else ""


def _mock_react_payload_from_messages(
    initial_payload: Dict[str, Any],
    parsed_user_messages: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    payload = copy.deepcopy(initial_payload)
    observations = [
        item.get("observation_record", item)
        for item in parsed_user_messages
        if item.get("task") == "tool_observation"
    ]
    payload["observations"] = observations
    payload["step"] = len(observations)
    return payload


def _mock_final_payload_from_messages(
    final_payload: Dict[str, Any],
    parsed_user_messages: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    payload = copy.deepcopy(final_payload)
    initial_react = next(
        (item for item in parsed_user_messages if item.get("task") == "react_tool_use"),
        {},
    )
    context = initial_react.get("context", {})
    if not isinstance(context, dict):
        context = {}
    tool_results = _mock_tool_results_from_observations(parsed_user_messages)
    task = payload.get("task")
    if task == "model_level_analysis":
        payload.setdefault("evaluation", context.get("evaluation", {}))
        payload.setdefault("tool_results", tool_results)
    elif task == "layer_level_analysis":
        payload.setdefault("model_candidates", context.get("model_candidates", {}))
        payload.setdefault("tool_results", tool_results)
    elif task == "solution_generation":
        for key in [
            "bottleneck_state",
            "current_hardware",
            "retrieved_cases",
            "simulator_schema",
            "search_space",
            "forbidden_hardware_fingerprints",
        ]:
            if key in context:
                payload.setdefault(key, context[key])
        payload.setdefault("tool_results", tool_results)
    return payload


def _mock_tool_results_from_observations(
    parsed_user_messages: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    tool_results: Dict[str, Dict[str, Any]] = {}
    for item in parsed_user_messages:
        if item.get("task") != "tool_observation":
            continue
        record = item.get("observation_record", {})
        if not isinstance(record, dict):
            continue
        result_key = record.get("result_key") or record.get("tool_name")
        observation = record.get("observation", {})
        if isinstance(result_key, str) and isinstance(observation, dict):
            tool_results[result_key] = {
                "payload": observation.get("payload", {}),
            }
    return tool_results


def _mock_react_tool_use(user_payload: Dict[str, Any]) -> Dict[str, Any]:
    agent_name = user_payload.get("agent_name")
    observations = user_payload.get("observations", [])
    called = {item.get("tool_name") for item in observations if isinstance(item, dict)}
    context = user_payload.get("context", {})
    solution_root = context.get("bottleneck_state", {}).get("dominant_root_cause")
    solution_update = _mock_direct_hardware_update(context, solution_root)
    sequences = {
        "model_level": [
            ("compare_search_states", {}),
            ("select_analysis_base", {"source": "current", "reason": "mock uses current evaluation as analysis base"}),
            ("summarize_metrics", {}),
            ("build_execution_timeline", {}),
            ("summarize_layer_rank_views", {"top_layers": 6}),
            ("summarize_operator_groups", {}),
            ("summarize_monetary_cost", {}),
            ("summarize_hardware_config", {}),
        ],
        "layer_level": [
            (
                "inspect_layer_details",
                {
                    "layers": [
                        {
                            "layer_id": item.get("layer_id"),
                            "layer_name": item.get("layer_name"),
                        }
                        for item in context.get("candidate_layers", [])[:8]
                    ]
                },
            ),
            ("summarize_operator_groups", {}),
            ("summarize_hardware_config", {}),
        ],
        "solution_generation": [
            ("summarize_hardware_config", {}),
            ("modify_hardware_parameter", solution_update),
        ],
    }
    for tool_name, arguments in sequences.get(str(agent_name), []):
        if tool_name not in called:
            return {
                "thought": f"Need {tool_name} evidence for {agent_name}.",
                "action": "tool",
                "tool": {"name": tool_name, "arguments": arguments},
            }
    return {
        "thought": f"Enough tool evidence gathered for {agent_name}.",
        "action": "finish",
        "finish_reason": "mock sequence complete",
    }


def _mock_direct_hardware_update(context: Dict[str, Any], root_cause: Any) -> Dict[str, Any]:
    from chiplet_tuner.core.search_space import (
        DEFAULT_HARDWARE_SEARCH_SPACE,
        step_chip_size,
        system_param_candidates,
    )
    from chiplet_tuner.core.utils import clamp_to_candidates

    hardware = context.get("current_hardware", {})
    search_space = context.get("search_space") or DEFAULT_HARDWARE_SEARCH_SPACE
    updates = []

    def add_system_step(parameter: str, direction: int) -> None:
        candidates = system_param_candidates(search_space, parameter)
        if parameter in hardware and candidates:
            updates.append(
                {
                    "parameter": parameter,
                    "value": clamp_to_candidates(int(hardware[parameter]), candidates, direction),
                }
            )

    if root_cause in {"compute", "memory", "buffer", "mixed"}:
        try:
            updates.append({"parameter": "chip_size", "value": step_chip_size(hardware, search_space, 1)})
        except ValueError:
            pass
    if root_cause == "compute":
        add_system_step("tensor_parall", 1)
    elif root_cause == "memory":
        add_system_step("dram_bw", 1)
        add_system_step("micro_batch", -1)
    elif root_cause == "communication":
        add_system_step("nop_bw", 1)
        add_system_step("tensor_parall", -1)
    elif root_cause == "buffer":
        add_system_step("micro_batch", -1)
    elif root_cause == "mixed":
        add_system_step("nop_bw", 1)
    if not updates:
        add_system_step("nop_bw", 1)
    return {"updates": updates or [{"parameter": "chiplet_type", "value": "os", "scope": "first"}]}


def _candidate_from_rank_item(item: Dict[str, Any], concern_type: str) -> Dict[str, Any]:
    timing = item.get("timing", {}) if isinstance(item.get("timing"), dict) else {}
    energy_record = item.get("energy", {}) if isinstance(item.get("energy"), dict) else {}
    latency_sum = float(item.get("latency_sum", timing.get("latency_sum", 0.0)) or 0.0)
    raw_energy = energy_record.get("total") if energy_record else item.get("energy", 0.0)
    if isinstance(raw_energy, dict):
        raw_energy = raw_energy.get("total", 0.0)
    if isinstance(raw_energy, dict):
        raw_energy = 0.0
    energy = float(raw_energy or 0.0)
    evidence = [
        f"{item.get('rank_metric')} rank value={float(item.get('rank_value', 0.0)):.4g}",
        f"latency_sum={latency_sum:.4g}",
        f"energy={energy:.4g}",
    ]
    return {
        "layer_id": item.get("layer_id", item.get("layerID")),
        "layer_name": item.get("layer_name", item.get("layerName")),
        "operator_group": item.get("group", item.get("operator_group", "unknown")),
        "concern_types": [concern_type],
        "rank_metric": item.get("rank_metric"),
        "rank_value": item.get("rank_value"),
        "evidence": evidence,
        "confidence": "high",
    }


def _mock_layer_level(user_payload: Dict[str, Any]) -> Dict[str, Any]:
    model_candidates = user_payload["model_candidates"]
    metrics = model_candidates["metrics"]
    tool_layers = (
        user_payload.get("tool_results", {})
        .get("inspect_layer_details", {})
        .get("payload", {})
        .get("layers", [])
    )
    if not tool_layers:
        tool_layers = model_candidates.get("candidate_layers", [])
    layer_diagnoses = []
    totals = {"compute": 0.0, "memory": 0.0, "communication": 0.0, "buffer": 0.0}
    for layer in tool_layers:
        ratios = _layer_dimension_scores(layer)
        dominant = max(ratios, key=ratios.get)
        for dim, ratio in ratios.items():
            totals[dim] += ratio
        impact_types = layer.get("impact_types") or ["latency"]
        layer_diagnoses.append(
            {
                "layerID": layer.get("layer_id", layer.get("layerID")),
                "layerName": layer.get("layer_name", layer.get("layerName")),
                "operator_group": layer.get("operator_group", layer.get("group")),
                "impact_types": impact_types,
                "root_causes": [dominant],
                "dominant_root_cause": dominant,
                "root_cause_ratios": {key: round(value, 4) for key, value in ratios.items()},
                "load_features": {
                    "occurrences": layer.get("placement", {}).get("occurrences", layer.get("occurrences")),
                    "cores": layer.get("placement", {}).get("cores", layer.get("cores", [])),
                    "batches": layer.get("placement", {}).get("batches", layer.get("batches", [])),
                    "latency_sum": layer.get("timing", {}).get("latency_sum", layer.get("latency_sum")),
                    "latency_max": layer.get("timing", {}).get("latency_max", layer.get("latency_max")),
                    "critical_end": layer.get("timing", {}).get("critical_end", layer.get("critical_end")),
                    "energy": layer.get("energy", {}).get("total", layer.get("energy")),
                    "features": layer.get("features", {}),
                },
                "diagnosis": f"{layer.get('operator_group', layer.get('group'))} layer is dominated by {dominant}.",
            }
        )
    if layer_diagnoses:
        totals = {key: value / len(layer_diagnoses) for key, value in totals.items()}
    dominant_root_cause = max(totals, key=totals.get) if layer_diagnoses else "unknown"
    focus = {
        "compute": ["chip_size", "tensor_parall"],
        "memory": ["dram_bw", "chip_size", "micro_batch"],
        "communication": ["nop_bw", "tensor_parall"],
        "buffer": ["chip_size", "chiplet_type"],
    }.get(dominant_root_cause, ["dram_bw", "nop_bw", "chip_size"])
    description = "; ".join(
        [
            "impact=latency",
            f"root_cause={dominant_root_cause}",
            f"latency={metrics.get('latency', 0.0):.4g}",
            f"energy={metrics.get('energy', 0.0):.4g}",
        ]
        + [
            "layer="
            f"{item['layerName']}|group={item['operator_group']}|root={item['dominant_root_cause']}|"
            f"compute={item['root_cause_ratios']['compute']:.3f}|memory={item['root_cause_ratios']['memory']:.3f}|"
            f"communication={item['root_cause_ratios']['communication']:.3f}|buffer={item['root_cause_ratios']['buffer']:.3f}"
            for item in layer_diagnoses[:5]
        ]
    )
    return {
        "primary_impact": "latency",
        "dominant_root_cause": dominant_root_cause,
        "layer_diagnoses": layer_diagnoses,
        "retrieval_description": description,
        "root_cause_summary": {key: round(value, 4) for key, value in totals.items()},
        "recommended_focus": focus,
        "notes": {"method": "mock_llm_ratio_analysis"},
    }


def _layer_dimension_scores(layer: Dict[str, Any]) -> Dict[str, float]:
    raw_scores = layer.get("root_cause_evidence", layer.get("dimension_scores"))
    if isinstance(raw_scores, dict):
        ratios = {}
        for key in ["compute", "memory", "communication", "buffer"]:
            try:
                ratios[key] = float(raw_scores.get(key, 0.0))
            except (TypeError, ValueError):
                ratios[key] = 0.0
        if any(ratios.values()):
            return ratios

    time_total = max(
        float(layer.get("timing", {}).get("calc_time", layer.get("calc_time", 0.0)))
        + float(layer.get("timing", {}).get("noc_time", layer.get("noc_time", 0.0)))
        + float(layer.get("timing", {}).get("dram_time", layer.get("dram_time", 0.0))),
        1.0,
    )
    energy_payload = layer.get("energy", {})
    if isinstance(energy_payload, dict):
        energy_total = max(float(energy_payload.get("total", 0.0)), 1.0)
        calc_energy = float(energy_payload.get("calc_energy", 0.0))
        dram_energy = float(energy_payload.get("dram_energy", 0.0))
        noc_energy = float(energy_payload.get("noc_energy", 0.0))
        ubuf_energy = float(energy_payload.get("ubuf_energy", 0.0))
    else:
        energy_total = max(float(layer.get("energy", 0.0)), 1.0)
        calc_energy = float(layer.get("calc_energy", 0.0))
        dram_energy = float(layer.get("dram_energy", 0.0))
        noc_energy = float(layer.get("noc_energy", 0.0))
        ubuf_energy = float(layer.get("ubuf_energy", 0.0))
    return {
        "compute": 0.5 * float(layer.get("timing", {}).get("calc_time", layer.get("calc_time", 0.0))) / time_total
        + 0.5 * calc_energy / energy_total,
        "memory": 0.5 * float(layer.get("timing", {}).get("dram_time", layer.get("dram_time", 0.0))) / time_total
        + 0.5 * dram_energy / energy_total,
        "communication": 0.5 * float(layer.get("timing", {}).get("noc_time", layer.get("noc_time", 0.0))) / time_total
        + 0.5 * noc_energy / energy_total,
        "buffer": ubuf_energy / energy_total,
    }


def _mock_solution(user_payload: Dict[str, Any]) -> Dict[str, Any]:
    btype = user_payload["bottleneck_state"]["dominant_root_cause"]
    tool_results = user_payload.get("tool_results", {})
    selected = "modify_hardware_parameter" if "modify_hardware_parameter" in tool_results else ""
    return {
        "strategy": f"mock_llm_{btype}",
        "hardware_update": _mock_hardware_update_payload(
            _mock_direct_hardware_update(
                {
                    "current_hardware": user_payload.get("current_hardware", {}),
                    "search_space": user_payload.get("search_space", {}),
                },
                btype,
            )
        ),
        "selected_hardware_candidate": selected,
        "actions": [],
        "rationale": f"Mock LLM proposal for {btype} bottleneck.",
        "notes": {"method": "mock_llm_solution"},
    }


def _mock_hardware_update_payload(tool_arguments: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    system_params: Dict[str, Any] = {}
    for update in tool_arguments.get("updates", []):
        parameter = update.get("parameter")
        value = update.get("value")
        if parameter == "chip_size":
            payload["chip_size"] = value
        elif parameter == "chiplet_type":
            payload["chiplet_type"] = value
        elif parameter in {"dram_bw", "nop_bw", "micro_batch", "tensor_parall"}:
            system_params[str(parameter)] = value
    if system_params:
        payload["system_params"] = system_params
    return payload


def create_llm_client(config: LLMConfig) -> LLMClient:
    provider = config.provider.lower()
    if provider == "mock":
        return MockLLMClient()
    if provider == "openai-compatible":
        return OpenAICompatibleClient(config)
    raise ValueError("Only mock and openai-compatible LLM clients are supported.")
