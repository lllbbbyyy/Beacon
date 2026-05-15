from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

from chiplet_tuner.agents.layer_level import LayerLevelAgent
from chiplet_tuner.agents.model_level import ModelLevelAgent
from chiplet_tuner.agents.solution import SolutionGenerationAgent
from chiplet_tuner.core.io import read_json
from chiplet_tuner.core.search_space import make_hardware_search_space
from chiplet_tuner.llm.clients import MockLLMClient
from chiplet_tuner.pipeline.tuner import MultiAgentTuner
from chiplet_tuner.rag.embeddings import create_embedding_model
from chiplet_tuner.rag.vector_store import HistoryVectorStore
from chiplet_tuner.simulators.base import GenericFileEvaluationAdapter
from chiplet_tuner.tools.analysis_tools import AnalysisToolbox


ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = ROOT / "Compass" / "try"
SOURCE_HARDWARE = SOURCE_RUN / "hardware_ws.json"
OUT_DIR = ROOT / "tmp" / "mock_prompt_dump"


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


class RecordingMockLLMClient(MockLLMClient):
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        super().__init__()

    def complete_json_messages(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        index = len(self.records)
        input_snapshot = copy.deepcopy(list(messages))
        effective_user_payload = self._last_task_payload(input_snapshot)
        task = str(effective_user_payload.get("task", "unknown"))
        agent_name = str(effective_user_payload.get("agent_name", "final"))
        response = super().complete_json_messages(input_snapshot)
        output_snapshot = copy.deepcopy(response)
        self.records.append(
            {
                "index": index,
                "agent_name": agent_name,
                "task": task,
                "system_prompt": "\n\n".join(
                    message["content"] for message in input_snapshot if message.get("role") == "system"
                ),
                "raw_input": {"messages": input_snapshot},
                "raw_output": output_snapshot,
            }
        )
        return response

    def _last_task_payload(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        parsed_user_payloads: List[Dict[str, Any]] = []
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            markdown_task = self._task_from_final_markdown(message.get("content", ""))
            if markdown_task:
                return {
                    "task": markdown_task,
                    "agent_name": "final",
                    "message_count": len(messages),
                }
            try:
                payload = json.loads(message.get("content", ""))
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

    def _task_from_final_markdown(self, content: str) -> str:
        match = re.search(r"Final (?:decision task|synthesis step):\s*`([^`]+)`", content)
        return match.group(1) if match else ""


def write_record_outputs(records: List[Dict[str, Any]]) -> None:
    assign_record_paths(records)
    for record in records:
        _write_json(OUT_DIR / record["input_json"], record["raw_input"])
        _write_json(OUT_DIR / record["output_json"], record["raw_output"])

    write_agent_dialogues(records)
    slim_records = [_slim_record(record) for record in records]
    prompt_calls_path = OUT_DIR / "prompt_calls.json"
    prompt_calls_path.write_text(json.dumps(slim_records, indent=2, ensure_ascii=False), encoding="utf-8")
    write_system_prompt_summary(records)
    write_readme(slim_records)


def _slim_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"raw_input", "raw_output", "system_prompt"}
    }


def assign_record_paths(records: List[Dict[str, Any]]) -> None:
    width = max(3, len(str(max(len(records) - 1, 0))))
    for record in records:
        display_index = f"{record['index']:0{width}d}"
        stem = f"{record['agent_name']}_{record['task']}"
        record["display_index"] = display_index
        record["input_json"] = f"raw_json/{display_index}_input_{stem}.json"
        record["output_json"] = f"raw_json/{display_index}_output_{stem}.json"


def write_system_prompt_summary(records: List[Dict[str, Any]]) -> None:
    seen: Dict[str, str] = {}
    sections = ["# Mock LLM System Prompts\n"]
    for record in records:
        system_prompt = record["system_prompt"]
        if system_prompt in seen:
            continue
        seen[system_prompt] = record["display_index"]
        sections.extend(
            [
                f"## call {record['display_index']}: {record['agent_name']} / {record['task']}\n",
            ]
        )
        _render_content_blocks(sections, system_prompt)
    _write_text(OUT_DIR / "system_prompts.md", "\n".join(sections))


def write_readme(records: List[Dict[str, Any]]) -> None:
    summary_lines = [
        "# Mock Prompt Dump",
        "",
        f"source_run: `{SOURCE_RUN}`",
        f"source_hardware: `{SOURCE_HARDWARE}`",
        "",
        "`raw_json/` contains the raw LLM input/output JSON for every call. `view_md/` contains one readable dialogue file per agent.",
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
        "| call | agent | task | raw input | raw output |",
        "|---:|---|---|---|---|",
    ]
    for record in records:
        summary_lines.append(
            f"| {record['display_index']} | {record['agent_name']} | {record['task']} | "
            f"`{record['input_json']}` | `{record['output_json']}` |"
        )
    _write_text(OUT_DIR / "README.md", "\n".join(summary_lines) + "\n")


def write_agent_dialogues(records: List[Dict[str, Any]]) -> None:
    for group in DIALOGUE_GROUPS:
        group_records = [
            record
            for record in records
            if record["agent_name"] == group["key"] or record["task"] == group["final_task"]
        ]
        if not group_records:
            continue
        _write_text(
            OUT_DIR / group["filename"],
            render_agent_dialogue_markdown(
                title=group["title"],
                final_task=group["final_task"],
                records=group_records,
            ),
        )


def render_agent_dialogue_markdown(
    title: str,
    final_task: str,
    records: List[Dict[str, Any]],
) -> str:
    final_records = [record for record in records if record["task"] == final_task]
    final_record = final_records[-1] if final_records else records[-1]
    messages = final_record["raw_input"].get("messages", [])
    lines = [
        f"# {title}",
        "",
        f"- final_call: `{final_record['display_index']}`",
        f"- final_task: `{final_record['task']}`",
        f"- message_count: `{len(messages)}`",
        f"- raw_input: `{final_record['input_json']}`",
        f"- raw_output: `{final_record['output_json']}`",
        "",
        "## Calls Included",
        "",
        "| call | task | raw input | raw output |",
        "|---:|---|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['display_index']} | {record['task']} | "
            f"`{record['input_json']}` | `{record['output_json']}` |"
        )
    lines.extend(["", "## Dialogue", ""])
    for index, message in enumerate(messages):
        role = message.get("role", "unknown")
        lines.extend([f"### Message {index:02d}: {role}", ""])
        _render_content_blocks(lines, message.get("content", ""))
    lines.extend([f"### Message {len(messages):02d}: assistant final output", ""])
    _render_json_block(lines, final_record["raw_output"])
    return "\n".join(lines)


def _render_content_blocks(lines: List[str], content: str) -> None:
    if str(content).lstrip().startswith(("Final decision task:", "Final synthesis step:")):
        lines.extend([str(content).strip(), ""])
        return
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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def reset_dump_outputs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for dirname in ["calls", "messages", "user_payloads", "raw_json", "view_md"]:
        shutil.rmtree(OUT_DIR / dirname, ignore_errors=True)
    for filename in ["prompt_calls.json", "system_prompts.md", "README.md"]:
        path = OUT_DIR / filename
        if path.exists():
            path.unlink()


def main() -> None:
    reset_dump_outputs()
    search_space = make_hardware_search_space(task_type="prefill", compute_scale="64")
    adapter = GenericFileEvaluationAdapter(
        run_dir=SOURCE_RUN,
        hardware_path=SOURCE_HARDWARE,
        search_space=search_space,
    )
    hardware = read_json(SOURCE_HARDWARE)
    evaluation = adapter.evaluate(hardware, iteration=0)

    toolbox = AnalysisToolbox()
    llm = RecordingMockLLMClient()
    history_store = HistoryVectorStore(
        OUT_DIR / "history.sqlite",
        embedding_model=create_embedding_model("hashing"),
    )
    simulator_schema = adapter.schema()
    tuner = MultiAgentTuner(
        model_agent=ModelLevelAgent(llm=llm, toolbox=toolbox),
        layer_agent=LayerLevelAgent(llm=llm, toolbox=toolbox),
        solution_agent=SolutionGenerationAgent(
            llm=llm,
            store=history_store,
            toolbox=toolbox,
            simulator_schema=simulator_schema,
            top_k=3,
        ),
        history_store=history_store,
        output_root=OUT_DIR,
        simulator=None,
    )
    tuner.analyze_evaluation(
        evaluation=evaluation,
        iteration=0,
        output_dir=OUT_DIR / "analysis_run",
    )
    write_record_outputs(llm.records)
    print(f"Wrote {len(llm.records)} prompt calls to {OUT_DIR}")


if __name__ == "__main__":
    main()
