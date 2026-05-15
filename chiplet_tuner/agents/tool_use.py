from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

from chiplet_tuner.core.schemas import ToolResult, ToolSpec
from chiplet_tuner.llm.clients import LLMClient
from chiplet_tuner.tools.analysis_tools import AnalysisToolbox, ToolContext


@dataclass
class ReactLoopResult:
    tool_results: Dict[str, ToolResult]
    transcript: List[Dict[str, Any]]
    messages: List[Dict[str, Any]]


def tool_specs_for_prompt(toolbox: AnalysisToolbox) -> List[Dict[str, Any]]:
    return [asdict(spec) if isinstance(spec, ToolSpec) else spec for spec in toolbox.specs()]


def execute_tool_plan(
    toolbox: AnalysisToolbox,
    tool_plan: Dict[str, Any],
    context: ToolContext,
) -> Dict[str, ToolResult]:
    tools = tool_plan.get("tools")
    if not isinstance(tools, list):
        raise ValueError("Tool plan must contain a 'tools' list.")
    results: Dict[str, ToolResult] = {}
    for item in tools:
        if not isinstance(item, dict):
            raise ValueError(f"Tool plan item must be an object, got {item!r}")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Tool plan item missing valid name: {item!r}")
        arguments = item.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError(f"Tool arguments for {name} must be an object.")
        results[name] = toolbox.run(name, context, arguments=arguments)
    return results


def serialize_tool_results(results: Dict[str, ToolResult]) -> Dict[str, Dict[str, Any]]:
    return {
        name: {
            "payload": _compact_payload(result.payload),
        }
        for name, result in results.items()
    }


def collect_generated_files(results: Dict[str, ToolResult]) -> Dict[str, str]:
    generated: Dict[str, str] = {}
    for result in results.values():
        generated.update(result.generated_files)
    return generated


def complete_json_continuing_messages(
    llm: LLMClient,
    messages: List[Dict[str, Any]],
    final_payload: Dict[str, Any],
    validate_response: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Dict[str, Any]:
    continued_messages = [dict(message) for message in messages]
    continued_messages.append({"role": "user", "content": _final_instruction_message(final_payload)})
    return complete_json_with_retries(
        llm,
        continued_messages,
        validate_response=validate_response,
        retry_context=_retry_context_from_payload(final_payload),
    )


def complete_json_with_retries(
    llm: LLMClient,
    messages: List[Dict[str, Any]],
    validate_response: Optional[Callable[[Dict[str, Any]], Any]] = None,
    retry_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call the LLM, retrying malformed JSON or schema-invalid responses.

    The function mutates ``messages`` when a retry is needed by appending an explicit
    correction request. This preserves the actual conversation that produced the final
    valid response, and every retry call is still captured by the normal LLM trace
    recorder.
    """

    max_retries = max(0, int(getattr(llm, "retry_attempts", 0) or 0))
    retry_temperature = float(getattr(llm, "retry_temperature", 0.1))
    context = retry_context or _retry_context_from_messages(messages)

    for attempt in range(max_retries + 1):
        attempt_response: Optional[Dict[str, Any]] = None
        previous_temperature = getattr(llm, "temperature_override", None)
        if attempt > 0:
            llm.temperature_override = retry_temperature
        try:
            response = llm.complete_json_messages(messages)
            attempt_response = response
            if validate_response is not None:
                validate_response(response)
            return response
        except Exception as exc:
            if not _is_retryable_llm_error(exc):
                raise
            if attempt >= max_retries:
                _record_retry_event(
                    llm,
                    context=context,
                    event="retry_exhausted",
                    retry_attempt=attempt,
                    max_retries=max_retries,
                    retry_temperature=retry_temperature,
                    error=exc,
                )
                raise
            retry_attempt = attempt + 1
            _record_retry_event(
                llm,
                context=context,
                event="retry",
                retry_attempt=retry_attempt,
                max_retries=max_retries,
                retry_temperature=retry_temperature,
                error=exc,
            )
            messages.append(
                {
                    "role": "user",
                    "content": _json_message(
                        _retry_payload(
                            context=context,
                            retry_attempt=retry_attempt,
                            max_retries=max_retries,
                            retry_temperature=retry_temperature,
                            error=exc,
                            previous_response=attempt_response,
                        )
                    ),
                }
            )
        finally:
            llm.temperature_override = previous_temperature

    raise RuntimeError("unreachable LLM retry state")


def _react_system_prompt(system_prompt: str, available_tools: List[Dict[str, Any]]) -> str:
    return (
        system_prompt.strip()
        + "\n\nAVAILABLE_TOOLS:\n"
        + json.dumps(available_tools, indent=2, ensure_ascii=False)
        + "\n\nThe AVAILABLE_TOOLS list above is authoritative. Tool descriptions and argument "
        "schemas are provided in this system message so tool-call behavior is not dependent "
        "on later user-message compression."
    )


def _json_message(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _final_instruction_message(payload: Dict[str, Any]) -> str:
    """Render the final decision request as readable prompt text.

    ReAct observations are already present in the message history.  The final turn
    should therefore read like an instruction to synthesize the final JSON answer,
    not like a machine envelope containing a prompt string.
    """

    task = str(payload.get("task", "final_decision"))
    instruction = str(payload.get("instruction", "")).strip()
    output_prompt = str(payload.get("output_prompt", "")).strip()
    structured_context = _compact_final_context({
        key: value
        for key, value in payload.items()
        if key not in {"task", "final_decision_required", "instruction", "output_prompt"}
    })
    sections = [
        f"Final synthesis step: `{task}`",
        "",
        _final_task_instruction(task, instruction),
        "",
        "Evidence source: use the prior messages in this conversation, especially the real tool observations. Tool use is closed; do not request additional tools.",
    ]
    if structured_context:
        sections.extend(
            [
                "",
                "Additional final-context metadata:",
                "```json",
                json.dumps(structured_context, indent=2, ensure_ascii=False, default=str),
                "```",
            ]
        )
    if output_prompt:
        sections.extend(
            [
                "",
                "Final output requirements:",
                output_prompt,
            ]
        )
    sections.extend(
        [
            "",
            "Return only the requested strict JSON object. Do not wrap it in Markdown.",
        ]
    )
    return "\n".join(sections)


def _final_task_instruction(task: str, fallback: str) -> str:
    instructions = {
        "model_level_analysis": "Produce the final model-level bottleneck candidate list for handoff to the layer-level agent.",
        "layer_level_analysis": "Produce the final layer-level bottleneck state for RAG retrieval and solution generation.",
        "solution_generation": "Produce the final legal hardware-update proposal for the next simulator evaluation.",
    }
    return instructions.get(task, fallback or "Produce the final strict JSON answer.")


def _compact_final_context(context: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key, value in context.items():
        if key == "active_analysis_base" and isinstance(value, dict):
            base = {
                name: value.get(name)
                for name in ["source", "base_key", "iteration", "selection_reason"]
                if value.get(name) is not None
            }
            if base:
                compact[key] = base
        else:
            compact[key] = _compact_any(value)
    return {key: value for key, value in compact.items() if value not in ({}, [], None)}


def _validate_react_response(response: Dict[str, Any], valid_tool_names: Optional[set[str]] = None) -> None:
    _require_str(response, "thought")
    action = _require_str(response, "action")
    if action == "finish":
        return
    if action != "tool":
        raise ValueError(f"ReAct action must be 'tool' or 'finish', got {action!r}")
    tool_call = response.get("tool")
    if not isinstance(tool_call, dict):
        raise ValueError("ReAct tool action must include object field 'tool'.")
    tool_name = _require_str(tool_call, "name")
    if valid_tool_names is not None and tool_name not in valid_tool_names:
        raise ValueError(f"Unknown tool {tool_name!r}; expected one of {sorted(valid_tool_names)}")
    arguments = tool_call.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("ReAct tool action field 'tool.arguments' must be an object.")


def _is_retryable_llm_error(exc: BaseException) -> bool:
    if isinstance(exc, json.JSONDecodeError):
        return True
    if isinstance(exc, KeyError):
        return True
    if isinstance(exc, ValueError):
        message = str(exc)
        if message.startswith("Unsupported chat message role") or message.startswith("Chat message content must be"):
            return False
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc)
        return (
            "LLM API returned non-JSON response" in message
            or "LLM response did not contain choices[0].message.content" in message
            or "LLM response content must be a string" in message
            or ("LLM request failed via OpenAI SDK" in message and "retryable=True" in message)
        )
    return False


def _record_retry_event(
    llm: LLMClient,
    context: Dict[str, Any],
    event: str,
    retry_attempt: int,
    max_retries: int,
    retry_temperature: float,
    error: BaseException,
) -> None:
    llm.record_retry_event(
        {
            "event": event,
            "task": context.get("task", "unknown"),
            "agent_name": context.get("agent_name", "unknown"),
            "retry_attempt": retry_attempt,
            "max_retry_attempts": max_retries,
            "retry_temperature": retry_temperature,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    )


def _retry_payload(
    context: Dict[str, Any],
    retry_attempt: int,
    max_retries: int,
    retry_temperature: float,
    error: BaseException,
    previous_response: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "task": "llm_retry_correction",
        "source_task": context.get("task", "unknown"),
        "agent_name": context.get("agent_name", "unknown"),
        "retry_attempt": retry_attempt,
        "max_retry_attempts": max_retries,
        "retry_temperature": retry_temperature,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "instruction": (
            "The previous LLM response was not usable by the framework. "
            "Keep the prior task context and real tool observations, and only correct the "
            "response format or missing fields. Return one strict JSON object only. "
            "Do not include markdown fences or explanatory text outside JSON."
        ),
        "required_response_schema": _retry_response_schema(
            task=str(context.get("task", "unknown")),
            agent_name=str(context.get("agent_name", "unknown")),
        ),
    }
    if previous_response is not None:
        payload["previous_parsed_response"] = previous_response
    return payload


def _retry_response_schema(task: str, agent_name: str) -> Dict[str, Any]:
    if task == "react_tool_use":
        return {
            "type": "object",
            "required": ["thought", "action"],
            "properties": {
                "thought": {"type": "string"},
                "action": {"type": "string", "enum": ["tool", "finish"]},
                "tool": {
                    "type": "object",
                    "required_when": {"action": "tool"},
                    "required": ["name", "arguments"],
                    "properties": {
                        "name": {"type": "string", "source": "AVAILABLE_TOOLS.name"},
                        "arguments": {"type": "object"},
                    },
                },
                "finish_reason": {"type": "string", "required_when": {"action": "finish"}},
            },
            "notes": [
                "Call at most one tool in this response.",
                "If action is finish, omit tool or leave it unused.",
            ],
        }
    if task == "model_level_analysis":
        return {
            "type": "object",
            "required": ["summary", "candidate_layers"],
            "properties": {
                "summary": {"type": "string"},
                "candidate_layers": {
                    "type": "array",
                    "min_items": 1,
                    "items": {
                        "type": "object",
                        "required": [
                            "layer_id",
                            "layer_name",
                            "operator_group",
                            "concern_types",
                            "rank_metric",
                            "rank_value",
                            "evidence",
                            "confidence",
                        ],
                        "properties": {
                            "layer_id": {"type": ["integer", "null"]},
                            "layer_name": {"type": ["string", "null"]},
                            "operator_group": {"type": "string"},
                            "concern_types": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["latency", "energy", "monetary_cost", "mixed"],
                                },
                            },
                            "rank_metric": {"type": ["string", "null"]},
                            "rank_value": {"type": ["number", "string", "null"]},
                            "evidence": {"type": "array", "items": {"type": "string"}},
                            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        },
                    },
                },
                "notes": {
                    "type": "object",
                    "properties": {
                        "selected_bottleneck_objective": {
                            "type": "string",
                            "enum": ["latency", "energy", "monetary_cost", "mixed", "unknown"],
                        },
                        "selected_rank_views": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "latency_sum",
                                    "energy",
                                    "critical_end",
                                    "compute",
                                    "memory",
                                    "communication",
                                    "buffer",
                                ],
                            },
                        },
                        "global_findings": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        }
    if task == "layer_level_analysis":
        impact_options = ["latency", "energy", "monetary_cost", "mixed", "unknown"]
        root_cause_options = [
            "compute",
            "memory",
            "communication",
            "buffer",
            "scheduling",
            "imbalance",
            "mixed",
            "unknown",
        ]
        ratio_object = {
            "type": "object",
            "required": ["compute", "memory", "communication", "buffer"],
            "properties": {
                "compute": {"type": "number"},
                "memory": {"type": "number"},
                "communication": {"type": "number"},
                "buffer": {"type": "number"},
            },
        }
        return {
            "type": "object",
            "required": [
                "primary_impact",
                "dominant_root_cause",
                "layer_diagnoses",
                "retrieval_description",
                "root_cause_summary",
                "recommended_focus",
            ],
            "properties": {
                "primary_impact": {"type": "string", "enum": impact_options},
                "dominant_root_cause": {"type": "string", "enum": root_cause_options},
                "layer_diagnoses": {
                    "type": "array",
                    "min_items": 1,
                    "items": {
                        "type": "object",
                        "required": [
                            "layerID",
                            "layerName",
                            "operator_group",
                            "impact_types",
                            "root_causes",
                            "dominant_root_cause",
                            "root_cause_ratios",
                            "load_features",
                            "diagnosis",
                        ],
                        "properties": {
                            "layerID": {"type": ["integer", "null"]},
                            "layerName": {"type": ["string", "null"]},
                            "operator_group": {"type": "string"},
                            "impact_types": {
                                "type": "array",
                                "items": {"type": "string", "enum": impact_options},
                            },
                            "root_causes": {
                                "type": "array",
                                "items": {"type": "string", "enum": root_cause_options},
                            },
                            "dominant_root_cause": {"type": "string", "enum": root_cause_options},
                            "root_cause_ratios": ratio_object,
                            "load_features": {"type": "object"},
                            "diagnosis": {"type": "string"},
                        },
                    },
                },
                "retrieval_description": {"type": "string"},
                "root_cause_summary": ratio_object,
                "recommended_focus": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "object"},
            },
        }
    if task == "solution_generation":
        return {
            "type": "object",
            "required": ["strategy", "actions", "rationale"],
            "any_of_required": [
                ["hardware_update"],
                ["selected_hardware_candidate"],
                ["hardware_candidate_key"],
                ["updated_hardware"],
            ],
            "properties": {
                "strategy": {"type": "string"},
                "hardware_update": {
                    "type": "object",
                    "properties": {
                        "chip_size": {"type": "integer", "source": "search_space.compute_spec candidates"},
                        "system_params": {
                            "type": "object",
                            "properties": {
                                "dram_bw": {"type": "number", "source": "search_space.system_params.dram_bw"},
                                "nop_bw": {"type": "number", "source": "search_space.system_params.nop_bw"},
                                "micro_batch": {
                                    "type": "integer",
                                    "source": "search_space.system_params.micro_batch",
                                },
                                "tensor_parall": {
                                    "type": "integer",
                                    "source": "search_space.system_params.tensor_parall",
                                },
                            },
                        },
                        "chiplet_type": {"type": "string", "source": "search_space.chiplet_type candidates"},
                        "chiplet_type_strategy": {
                            "type": "string",
                            "enum": ["preserve_prefix", "majority", "uniform"],
                        },
                        "chiplet_type_fill": {
                            "type": "string",
                            "source": "search_space.chiplet_type candidates",
                        },
                    },
                },
                "selected_hardware_candidate": {"type": "string"},
                "hardware_candidate_key": {"type": "string"},
                "updated_hardware": {"type": "object"},
                "actions": {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string"},
                "notes": {
                    "type": "object",
                    "properties": {
                        "history_usage": {"type": "string"},
                        "risk": {"type": "string"},
                    },
                },
            },
            "notes": [
                "Do not hand-edit derived hardware fields when a hardware_update intent is sufficient.",
                "Use simulator field name tensor_parall, not tensor_parallel.",
            ],
        }
    return {
        "type": "object",
        "notes": [
            f"No task-specific retry schema is registered for task={task!r}, agent={agent_name!r}.",
            "Return the strict JSON object requested by the previous task prompt.",
        ],
    }


def _retry_context_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    task = str(payload.get("task", "unknown"))
    return {"task": task, "agent_name": _agent_name_from_task(task)}


def _retry_context_from_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        task = str(payload.get("task", "unknown"))
        agent_name = payload.get("agent_name")
        if not isinstance(agent_name, str) or not agent_name:
            agent_name = _agent_name_from_task(task)
        return {"task": task, "agent_name": agent_name}
    return {"task": "unknown", "agent_name": "unknown"}


def _agent_name_from_task(task: str) -> str:
    if task.startswith("model_level"):
        return "model_level"
    if task.startswith("layer_level"):
        return "layer_level"
    if task.startswith("solution"):
        return "solution_generation"
    if task == "react_tool_use":
        return "react"
    return "unknown"


def _require_str(payload: Dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ReAct response must contain non-empty string field {key!r}")
    return value


def _unique_result_key(results: Dict[str, ToolResult], tool_name: str) -> str:
    if tool_name not in results:
        return tool_name
    index = 2
    while f"{tool_name}#{index}" in results:
        index += 1
    return f"{tool_name}#{index}"


def _tool_call_key(tool_name: str, arguments: Dict[str, Any]) -> str:
    return f"{tool_name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"


def _full_tool_result(result: ToolResult) -> Dict[str, Any]:
    payload = _compact_payload(result.payload)
    compact = {
        "name": result.name,
        "payload": payload,
    }
    if result.generated_files:
        compact["generated_artifacts"] = sorted(result.generated_files)
    return compact


def _compact_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key, value in payload.items():
        if key in {"run_dir", "hardware_fingerprint", "updated_hardware_fingerprint"}:
            continue
        if key == "updated_hardware" and isinstance(value, dict):
            compact[key] = _compact_hardware(value)
        elif key == "layer_loads" and isinstance(value, list):
            compact[key] = {"count": len(value), "head": [_compact_layer_like(item) for item in value[:5] if isinstance(item, dict)]}
        elif key == "bottleneck_layers" and isinstance(value, list):
            compact[key] = [_compact_layer_like(item) if isinstance(item, dict) else item for item in value[:8]]
        elif isinstance(value, list) and len(value) > 10:
            compact[key] = {"count": len(value), "head": [_compact_any(item) for item in value[:10]]}
        elif isinstance(value, dict) and len(value) > 20:
            compact[key] = {"keys": sorted(value)[:20], "size": len(value)}
        else:
            compact[key] = _compact_any(value)
    return compact


def _compact_any(value: Any) -> Any:
    if isinstance(value, dict):
        if "layer_id" in value or "layerID" in value:
            return _compact_layer_like(value)
        if "chiplets" in value:
            return _compact_hardware(value)
        skipped = {"run_dir", "hardware_fingerprint", "updated_hardware_fingerprint"}
        return {key: _compact_any(item) for key, item in value.items() if key not in skipped}
    if isinstance(value, list):
        if len(value) > 10:
            return {"count": len(value), "head": [_compact_any(item) for item in value[:10]]}
        return [_compact_any(item) for item in value]
    return value


def _compact_hardware(value: Dict[str, Any]) -> Dict[str, Any]:
    chiplets = value.get("chiplets", [])
    type_counts: Dict[str, int] = {}
    compute_counts: Dict[str, int] = {}
    buffer_counts: Dict[str, int] = {}
    if isinstance(chiplets, list):
        for chip in chiplets:
            if not isinstance(chip, dict):
                continue
            type_counts[str(chip.get("type"))] = type_counts.get(str(chip.get("type")), 0) + 1
            compute_counts[str(chip.get("compute_units"))] = compute_counts.get(str(chip.get("compute_units")), 0) + 1
            buffer_counts[str(chip.get("buffer_size"))] = buffer_counts.get(str(chip.get("buffer_size")), 0) + 1
    return {
        "num_chiplets": value.get("num_chiplets"),
        "chip_x": value.get("chip_x"),
        "chip_y": value.get("chip_y"),
        "dram_bw": value.get("dram_bw"),
        "nop_bw": value.get("nop_bw"),
        "micro_batch": value.get("micro_batch"),
        "tensor_parall": value.get("tensor_parall"),
        "chiplet_count": len(chiplets) if isinstance(chiplets, list) else None,
        "chiplet_types": type_counts,
        "compute_units": compute_counts,
        "buffer_sizes": buffer_counts,
    }


def _compact_layer_like(value: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "layer_id",
        "layer_name",
        "operator_group",
        "group",
        "rank_metric",
        "rank_value",
        "latency_sum",
        "energy",
        "critical_end",
        "dominant_dimension",
        "dominant_root_cause",
        "impact_types",
        "root_causes",
        "shares",
        "timing",
        "breakdown",
        "root_cause",
    ]
    compact = {key: _compact_any(value[key]) for key in keep if key in value}
    if "layer_id" not in compact and "layerID" in value:
        compact["layer_id"] = value.get("layerID")
    if "layer_name" not in compact and "layerName" in value:
        compact["layer_name"] = value.get("layerName")
    return compact
