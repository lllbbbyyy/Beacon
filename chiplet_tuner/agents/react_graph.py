from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from chiplet_tuner.agents.tool_use import (
    ReactLoopResult,
    _full_tool_result,
    _json_message,
    _react_system_prompt,
    _require_str,
    _tool_call_key,
    _unique_result_key,
    _validate_react_response,
    complete_json_with_retries,
    tool_specs_for_prompt,
)
from chiplet_tuner.core.schemas import ToolResult
from chiplet_tuner.llm.clients import LLMClient
from chiplet_tuner.tools.analysis_tools import AnalysisToolbox, ToolContext


class ReactGraphState(TypedDict, total=False):
    messages: List[Dict[str, Any]]
    tool_results: Dict[str, Dict[str, Any]]
    transcript: List[Dict[str, Any]]
    seen_tool_calls: List[str]
    step: int
    max_steps: int
    agent_name: str
    pending_response: Dict[str, Any]
    pending_assistant_message: Dict[str, Any]
    pending_tool_name: str
    pending_tool_arguments: Dict[str, Any]
    pending_duplicate: bool
    finished: bool


def run_react_tool_graph(
    llm: LLMClient,
    toolbox: AnalysisToolbox,
    system_prompt: str,
    task: str,
    agent_name: str,
    agent_goal: str,
    context_payload: Dict[str, Any],
    context: ToolContext,
    max_steps: int = 8,
) -> ReactLoopResult:
    """Run a generic LangGraph ReAct loop for one agent.

    The graph is intentionally agent-agnostic: agent role, prompts, context, and
    final schema remain outside the tool loop. This keeps tool execution,
    duplicate-tool handling, retry behavior, and observations consistent across
    model-level, layer-level, and solution-generation agents.
    """

    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")

    graph = _build_react_graph(llm=llm, toolbox=toolbox, context=context, task=task)
    messages = [
        {
            "role": "system",
            "content": _react_system_prompt(
                system_prompt=system_prompt,
                available_tools=tool_specs_for_prompt(toolbox),
            ),
        },
        {
            "role": "user",
            "content": _json_message(
                {
                    "task": task,
                    "agent_name": agent_name,
                    "agent_goal": agent_goal,
                    "context": context_payload,
                    "step": 0,
                    "max_steps": max_steps,
                    "instruction": "Choose exactly one next tool call or finish tool use.",
                }
            ),
        },
    ]
    initial_state: ReactGraphState = {
        "messages": messages,
        "tool_results": {},
        "transcript": [],
        "seen_tool_calls": [],
        "step": 0,
        "max_steps": max_steps,
        "agent_name": agent_name,
        "finished": False,
    }
    final_state = graph.compile().invoke(
        initial_state,
        config={"recursion_limit": max(10, max_steps * 4 + 5)},
    )
    return ReactLoopResult(
        tool_results=_tool_results_from_records(final_state.get("tool_results", {})),
        transcript=final_state.get("transcript", []),
        messages=final_state.get("messages", messages),
    )


def _tool_result_record(result: ToolResult) -> Dict[str, Any]:
    return {
        "name": result.name,
        "payload": result.payload,
        "generated_files": result.generated_files,
    }


def _tool_result_from_record(record: Dict[str, Any]) -> ToolResult:
    return ToolResult(
        name=str(record.get("name", "")),
        payload=dict(record.get("payload", {})),
        generated_files=dict(record.get("generated_files", {})),
    )


def _tool_results_from_records(records: Dict[str, Dict[str, Any]]) -> Dict[str, ToolResult]:
    return {key: _tool_result_from_record(record) for key, record in records.items()}


def _build_react_graph(
    llm: LLMClient,
    toolbox: AnalysisToolbox,
    context: ToolContext,
    task: str,
) -> Any:
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise ImportError(
            "LangGraph is required for agent ReAct tool execution. "
            "Install it in the active environment with: pip install langgraph"
        ) from exc

    valid_tool_names = {spec.name for spec in toolbox.specs()}

    def llm_decide(state: ReactGraphState) -> Dict[str, Any]:
        step = int(state.get("step", 0))
        max_steps = int(state.get("max_steps", 0))
        agent = str(state.get("agent_name", "unknown"))
        if step >= max_steps:
            transcript = list(state.get("transcript", []))
            transcript.append(
                {
                    "step": max_steps,
                    "thought": f"{agent} reached max_steps={max_steps}; continuing with gathered evidence.",
                    "action": "finish",
                    "finish_reason": "max_steps reached without explicit finish action",
                }
            )
            return {"finished": True, "transcript": transcript}

        messages = list(state.get("messages", []))
        response = complete_json_with_retries(
            llm,
            messages,
            validate_response=lambda payload: _validate_react_response(payload, valid_tool_names),
            retry_context={"task": task, "agent_name": agent},
        )
        assistant_message = llm.assistant_message_from_response(response, _json_message(response))
        action = _require_str(response, "action")
        thought = _require_str(response, "thought")
        if action == "finish":
            messages.append(assistant_message)
            transcript = list(state.get("transcript", []))
            transcript.append(
                {
                    "step": step,
                    "thought": thought,
                    "action": "finish",
                    "finish_reason": response.get("finish_reason", ""),
                }
            )
            return {"messages": messages, "transcript": transcript, "finished": True}

        tool_call = response.get("tool")
        if not isinstance(tool_call, dict):
            raise ValueError("ReAct tool action must include object field 'tool'.")
        tool_name = _require_str(tool_call, "name")
        arguments = tool_call.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError(f"Tool arguments for {tool_name} must be an object.")
        duplicate = _tool_call_key(tool_name, arguments) in set(state.get("seen_tool_calls", []))
        return {
            "pending_response": response,
            "pending_assistant_message": assistant_message,
            "pending_tool_name": tool_name,
            "pending_tool_arguments": arguments,
            "pending_duplicate": duplicate,
            "finished": False,
        }

    def execute_tool(state: ReactGraphState) -> Dict[str, Any]:
        step = int(state.get("step", 0))
        max_steps = int(state.get("max_steps", 0))
        agent = str(state.get("agent_name", "unknown"))
        messages = list(state.get("messages", []))
        transcript = list(state.get("transcript", []))
        tool_results = dict(state.get("tool_results", {}))
        seen_tool_calls = set(state.get("seen_tool_calls", []))
        response = state["pending_response"]
        assistant_message = state["pending_assistant_message"]
        tool_name = state["pending_tool_name"]
        arguments = state.get("pending_tool_arguments", {})
        thought = _require_str(response, "thought")

        if state.get("pending_duplicate"):
            messages.append(assistant_message)
            transcript.append(
                {
                    "step": step,
                    "thought": thought,
                    "action": "finish",
                    "finish_reason": f"duplicate tool call {tool_name}; continuing with gathered evidence",
                }
            )
            return {"messages": messages, "transcript": transcript, "finished": True}

        seen_tool_calls.add(_tool_call_key(tool_name, arguments))
        result = toolbox.run(tool_name, context, arguments=arguments)
        result_key = _unique_result_key(tool_results, tool_name)
        tool_results[result_key] = _tool_result_record(result)
        observation = {
            "step": step,
            "tool_name": tool_name,
            "result_key": result_key,
            "arguments": arguments,
            "observation": _full_tool_result(result),
        }
        messages.append(assistant_message)
        messages.append(
            {
                "role": "user",
                "content": _json_message(
                    {
                        "task": "tool_observation",
                        "agent_name": agent,
                        "observation_record": observation,
                        "next_step": step + 1,
                        "max_steps": max_steps,
                        "instruction": (
                            "Use this real tool observation with the previous messages. "
                            "Choose exactly one next tool call or finish tool use."
                        ),
                    }
                ),
            }
        )
        transcript.append(
            {
                "step": step,
                "thought": thought,
                "action": "tool",
                "tool_name": tool_name,
                "result_key": result_key,
                "arguments": arguments,
            }
        )
        return {
            "messages": messages,
            "tool_results": tool_results,
            "transcript": transcript,
            "seen_tool_calls": sorted(seen_tool_calls),
            "step": step + 1,
            "pending_response": {},
            "pending_assistant_message": {},
            "pending_tool_name": "",
            "pending_tool_arguments": {},
            "pending_duplicate": False,
        }

    def route_after_decide(state: ReactGraphState) -> str:
        if state.get("finished"):
            return "end"
        return "execute_tool"

    def route_after_tool(state: ReactGraphState) -> str:
        if state.get("finished"):
            return "end"
        return "llm_decide"

    graph = StateGraph(ReactGraphState)
    graph.add_node("llm_decide", llm_decide)
    graph.add_node("execute_tool", execute_tool)
    graph.set_entry_point("llm_decide")
    graph.add_conditional_edges(
        "llm_decide",
        route_after_decide,
        {
            "execute_tool": "execute_tool",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "execute_tool",
        route_after_tool,
        {
            "llm_decide": "llm_decide",
            "end": END,
        },
    )
    return graph
