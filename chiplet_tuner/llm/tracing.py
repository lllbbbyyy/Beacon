from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


DIALOGUE_GROUPS = [
    {
        "key": "model_level",
        "title": "Model-Level Agent Dialogue",
        "filename": "view_md/model_level_agent_dialogue.md",
        "final_task": "model_level_analysis",
    },
    {
        "key": "layer_level",
        "title": "Layer-Level Agent Dialogue",
        "filename": "view_md/layer_level_agent_dialogue.md",
        "final_task": "layer_level_analysis",
    },
    {
        "key": "solution_generation",
        "title": "Solution-Generation Agent Dialogue",
        "filename": "view_md/solution_generation_agent_dialogue.md",
        "final_task": "solution_generation",
    },
]

FINAL_TASK_TO_AGENT = {
    "model_level_analysis": "model_level",
    "layer_level_analysis": "layer_level",
    "solution_generation": "solution_generation",
}


class LLMTraceRecorder:
    """Incrementally persist raw LLM request/response data and readable dialogues."""

    def __init__(self, output_dir: Path, index_width: int = 4) -> None:
        self.output_dir = Path(output_dir)
        self.index_width = index_width
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "raw_json").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "view_md").mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = self._load_existing_records()

    def begin_call(self, request_body: Dict[str, Any]) -> int:
        index = len(self.records)
        input_snapshot = copy.deepcopy(request_body)
        messages = input_snapshot.get("messages")
        if messages is None and isinstance(input_snapshot.get("request_body"), dict):
            messages = input_snapshot["request_body"].get("messages")
        if not isinstance(messages, list):
            messages = []
        effective_payload = self._last_task_payload(messages)
        task = str(effective_payload.get("task", "unknown"))
        agent_name = str(
            effective_payload.get("agent_name")
            or FINAL_TASK_TO_AGENT.get(task)
            or "unknown"
        )
        display_index = f"{index:0{self.index_width}d}"
        stem = task if task.startswith(agent_name) else f"{agent_name}_{task}"
        record = {
            "index": index,
            "display_index": display_index,
            "agent_name": agent_name,
            "task": task,
            "status": "pending",
            "started_at": _utc_now_iso(),
            "started_at_monotonic": time.monotonic(),
            "ended_at": None,
            "duration_s": None,
            "message_count": len(messages),
            "system_prompt": "\n\n".join(
                message["content"]
                for message in messages
                if isinstance(message, dict) and message.get("role") == "system"
            ),
            "raw_input": input_snapshot,
            "raw_output": None,
            "response_content": None,
            "error": None,
            "input_json": f"raw_json/{display_index}_input_{stem}.json",
            "output_json": f"raw_json/{display_index}_output_{stem}.json",
        }
        self.records.append(record)
        self._write_json(self.output_dir / record["input_json"], input_snapshot)
        self._write_summary_files()
        return index

    def end_call(
        self,
        call_index: int,
        raw_output: Dict[str, Any],
        response_content: Optional[str] = None,
        error: Optional[BaseException | str] = None,
    ) -> None:
        record = self.records[call_index]
        output_snapshot = copy.deepcopy(raw_output)
        started_at = record.get("started_at_monotonic")
        if isinstance(started_at, (int, float)):
            record["duration_s"] = round(time.monotonic() - float(started_at), 6)
        record["ended_at"] = _utc_now_iso()
        record["raw_output"] = output_snapshot
        record["usage"] = _extract_usage(output_snapshot)
        record["response_content"] = response_content
        if error is None:
            record["status"] = "ok"
        else:
            record["status"] = "error"
            record["error"] = _error_payload(error)
        self._write_json(self.output_dir / record["output_json"], output_snapshot)
        self._write_summary_files()

    def _last_task_payload(self, messages: Sequence[Any]) -> Dict[str, Any]:
        parsed_user_payloads: List[Dict[str, Any]] = []
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            try:
                payload = json.loads(str(message.get("content", "")))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                parsed_user_payloads.append(payload)
        parsed_user_payloads = list(reversed(parsed_user_payloads))
        latest_payload = parsed_user_payloads[-1] if parsed_user_payloads else {}
        latest_task = latest_payload.get("task")
        if latest_task and latest_task != "tool_observation":
            effective = dict(latest_payload)
            effective["message_count"] = len(messages)
            return effective
        react_payload = next(
            (payload for payload in parsed_user_payloads if payload.get("task") == "react_tool_use"),
            None,
        )
        if react_payload is not None:
            effective = dict(react_payload)
            effective["message_count"] = len(messages)
            effective["latest_user_message"] = latest_payload
            return effective
        for payload in reversed(parsed_user_payloads):
            if payload.get("task"):
                return payload
        return {}

    def _write_summary_files(self) -> None:
        slim_records = [self._slim_record(record) for record in self.records]
        self._write_json(self.output_dir / "prompt_calls.json", slim_records)
        self._write_system_prompt_summary()
        self._write_agent_dialogues()
        self._write_readme(slim_records)

    def _slim_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in record.items()
            if key
            not in {
                "raw_input",
                "raw_output",
                "system_prompt",
                "response_content",
                "started_at_monotonic",
            }
        }

    def _write_system_prompt_summary(self) -> None:
        seen: Dict[str, str] = {}
        sections = ["# LLM System Prompts", ""]
        for record in self.records:
            system_prompt = record["system_prompt"]
            if not system_prompt or system_prompt in seen:
                continue
            seen[system_prompt] = record["display_index"]
            sections.extend(
                [
                    f"## call {record['display_index']}: {record['agent_name']} / {record['task']}",
                    "",
                ]
            )
            _render_content_blocks(sections, system_prompt)
        self._write_text(self.output_dir / "system_prompts.md", "\n".join(sections).rstrip() + "\n")

    def _write_agent_dialogues(self) -> None:
        for group in DIALOGUE_GROUPS:
            group_records = [
                record
                for record in self.records
                if record["agent_name"] == group["key"] or record["task"] == group["final_task"]
            ]
            if not group_records:
                continue
            self._write_text(
                self.output_dir / group["filename"],
                render_agent_dialogue_markdown(
                    title=group["title"],
                    final_task=group["final_task"],
                    records=group_records,
                ),
            )

    def _write_readme(self, records: List[Dict[str, Any]]) -> None:
        lines = [
            "# LLM Trace",
            "",
            "`raw_json/` stores raw request/response JSON for every LLM call. "
            "`view_md/` stores readable per-agent dialogue expansions.",
            "",
            "## Readable Dialogues",
            "",
            "| agent | dialogue |",
            "|---|---|",
            "| model_level | `view_md/model_level_agent_dialogue.md` |",
            "| layer_level | `view_md/layer_level_agent_dialogue.md` |",
            "| solution_generation | `view_md/solution_generation_agent_dialogue.md` |",
            "",
            "## Raw Calls",
            "",
            "| call | status | agent | task | duration_s | raw input | raw output |",
            "|---:|---|---|---|---:|---|---|",
        ]
        for record in records:
            lines.append(
                f"| {record['display_index']} | {record['status']} | {record['agent_name']} | {record['task']} | "
                f"{_format_duration_cell(record.get('duration_s'))} | "
                f"`{record['input_json']}` | `{record['output_json']}` |"
            )
        self._write_text(self.output_dir / "README.md", "\n".join(lines) + "\n")

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _load_existing_records(self) -> List[Dict[str, Any]]:
        calls_path = self.output_dir / "prompt_calls.json"
        if not calls_path.exists():
            return []
        try:
            slim_records = json.loads(calls_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(slim_records, list):
            return []
        records: List[Dict[str, Any]] = []
        for item in slim_records:
            if not isinstance(item, dict):
                continue
            input_json = str(item.get("input_json", ""))
            output_json = str(item.get("output_json", ""))
            raw_input = self._read_existing_json(input_json)
            raw_output = self._read_existing_json(output_json)
            messages = _messages_from_raw_input(raw_input) if isinstance(raw_input, dict) else []
            system_prompt = "\n\n".join(
                message["content"]
                for message in messages
                if isinstance(message, dict) and message.get("role") == "system"
            )
            response_content = None
            if isinstance(raw_output, dict) and isinstance(raw_output.get("content"), str):
                response_content = raw_output["content"]
            records.append(
                {
                    **item,
                    "started_at_monotonic": None,
                    "system_prompt": system_prompt,
                    "raw_input": raw_input if isinstance(raw_input, dict) else {},
                    "raw_output": raw_output if isinstance(raw_output, dict) else None,
                    "response_content": response_content,
                }
            )
        return records

    def _read_existing_json(self, relative_path: str) -> Any:
        if not relative_path:
            return None
        path = self.output_dir / relative_path
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None


def render_agent_dialogue_markdown(
    title: str,
    final_task: str,
    records: List[Dict[str, Any]],
) -> str:
    completed_dialogue_records = [
        record
        for record in records
        if record["task"] == final_task or record["status"] == "error"
    ]
    if not completed_dialogue_records:
        completed_dialogue_records = [records[-1]]
    lines = [
        f"# {title}",
        "",
        "## Calls Included",
        "",
        "| call | status | task | raw input | raw output |",
        "|---:|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['display_index']} | {record['status']} | {record['task']} | "
            f"`{record['input_json']}` | `{record['output_json']}` |"
        )
    for dialogue_index, record in enumerate(completed_dialogue_records, start=1):
        messages = _messages_from_raw_input(record["raw_input"])
        if not isinstance(messages, list):
            messages = []
        lines.extend(
            [
                "",
                f"## Dialogue {dialogue_index}: call {record['display_index']}",
                "",
                f"- task: `{record['task']}`",
                f"- status: `{record['status']}`",
                f"- message_count: `{len(messages)}`",
                f"- raw_input: `{record['input_json']}`",
                f"- raw_output: `{record['output_json']}`",
                "",
            ]
        )
        if record.get("error"):
            lines.extend(["### Error", ""])
            _render_json_block(lines, record["error"])
        lines.extend(["### Dialogue Messages", ""])
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("role", "unknown")
            lines.extend([f"#### Message {message_index:02d}: {role}", ""])
            _render_content_blocks(lines, str(message.get("content", "")))
        lines.extend([f"#### Message {len(messages):02d}: assistant output", ""])
        response_content = record.get("response_content")
        if isinstance(response_content, str) and response_content.strip():
            _render_content_blocks(lines, response_content)
        else:
            _render_json_block(lines, record.get("raw_output"))
    return "\n".join(lines).rstrip() + "\n"


def _render_content_blocks(lines: List[str], content: str) -> None:
    fragments = _split_json_fragments(content)
    for kind, value in fragments:
        if kind == "json":
            _render_json_block(lines, value)
        else:
            text = str(value).strip()
            if text:
                lines.extend(["```text", text, "```", ""])


def _split_json_fragments(content: str) -> List[tuple[str, Any]]:
    decoder = json.JSONDecoder()
    fragments: List[tuple[str, Any]] = []
    text_start = 0
    index = 0
    while index < len(content):
        if content[index] not in "{[":
            index += 1
            continue
        try:
            parsed, end = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            index += 1
            continue
        if text_start < index:
            fragments.append(("text", content[text_start:index]))
        fragments.append(("json", parsed))
        index += end
        text_start = index
    if text_start < len(content):
        fragments.append(("text", content[text_start:]))
    return fragments or [("text", content)]


def _render_json_block(lines: List[str], payload: Any) -> None:
    lines.extend(["```json", json.dumps(payload, indent=2, ensure_ascii=False), "```", ""])


def _messages_from_raw_input(raw_input: Dict[str, Any]) -> List[Any]:
    messages = raw_input.get("messages")
    if messages is None and isinstance(raw_input.get("request_body"), dict):
        messages = raw_input["request_body"].get("messages")
    if isinstance(messages, list):
        return messages
    return []


def _error_payload(error: BaseException | str) -> Dict[str, str]:
    if isinstance(error, BaseException):
        return {"type": type(error).__name__, "message": str(error)}
    return {"type": "Error", "message": str(error)}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _extract_usage(raw_output: Dict[str, Any]) -> Dict[str, Any]:
    response = raw_output.get("response") if isinstance(raw_output, dict) else None
    if not isinstance(response, dict):
        return {}
    usage = response.get("usage")
    return copy.deepcopy(usage) if isinstance(usage, dict) else {}


def _format_duration_cell(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"
