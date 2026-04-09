from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import GenerationConfig
from .model_client import request_model
from .parser import GoogleSearchParser, ToolCall, extract_answer_block, truncate_at_tool_call
from .prompts import SYSTEM_PROMPT, USER_PROMPT
from .search import search


class AgentState(Enum):
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


@dataclass(slots=True)
class AgentData:
    messages: list[dict[str, str]]
    user_turns: int = 0
    assistant_turns: int = 0
    total_response_length: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    init_messages_length: int = 0

    def __post_init__(self) -> None:
        self.init_messages_length = len(self.messages)


class ToolAgentLoop:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.tool_parser = GoogleSearchParser()

    def run(self, record: dict[str, Any]) -> tuple[str, str]:
        if self._use_direct_mode():
            return self._run_direct(record)
        return self._run_agent(record)

    def _use_direct_mode(self) -> bool:
        if self.config.mode == "direct":
            return True
        if self.config.mode == "agent":
            return False
        return self.config.model_name.strip().lower() == "qwen3-8b"

    def _run_direct(self, record: dict[str, Any]) -> tuple[str, str]:
        messages = [{"role": "user", "content": USER_PROMPT.format(query=record["query"])}]
        full_response = request_model(
            self.config.base_url,
            self.config.model_name,
            messages,
        )
        return extract_answer_block(full_response), full_response

    def _run_agent(self, record: dict[str, Any]) -> tuple[str, str]:
        agent_data = AgentData(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(query=record["query"])},
            ]
        )

        state = AgentState.GENERATING
        while state != AgentState.TERMINATED:
            if state == AgentState.GENERATING:
                state = self._handle_generating_state(agent_data)
            elif state == AgentState.PROCESSING_TOOLS:
                state = self._handle_processing_tools_state(agent_data)
            else:
                state = AgentState.TERMINATED

        response_messages = agent_data.messages[agent_data.init_messages_length :]
        full_response = "\n".join(message["content"] for message in response_messages).strip()
        return extract_answer_block(full_response), full_response

    def _response_length(self, text: str) -> int:
        return len(text.split())

    def _handle_generating_state(self, agent_data: AgentData) -> AgentState:
        output = request_model(
            self.config.base_url,
            self.config.model_name,
            agent_data.messages,
            extra_body={
                "stop": ["</google_search>"],
                "include_stop_str_in_output": True,
            },
        )

        should_truncate = True
        if (
            self.config.max_assistant_turns is not None
            and agent_data.assistant_turns + 1 >= self.config.max_assistant_turns
        ):
            _, tool_calls = self.tool_parser.extract_tool_calls(output)
            if tool_calls:
                output += "\n<answer>Cannot determine an answer based on the available information.</answer>"
                should_truncate = False

        if should_truncate:
            output = truncate_at_tool_call(output)

        projected_length = agent_data.total_response_length + self._response_length(output)
        if (
            self.config.max_response_length is not None
            and projected_length >= self.config.max_response_length
        ):
            return AgentState.TERMINATED

        if (
            self.config.max_assistant_turns is not None
            and agent_data.assistant_turns + 1 > self.config.max_assistant_turns
        ):
            return AgentState.TERMINATED

        if (
            self.config.max_user_turns is not None
            and agent_data.user_turns > self.config.max_user_turns
        ):
            return AgentState.TERMINATED

        agent_data.assistant_turns += 1
        agent_data.total_response_length = projected_length
        agent_data.messages.append({"role": "assistant", "content": output})
        _, agent_data.tool_calls = self.tool_parser.extract_tool_calls(output)

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        return AgentState.TERMINATED

    def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        for tool_call in agent_data.tool_calls:
            response = self._call_tool(tool_call)
            agent_data.total_response_length += self._response_length(response)
            if (
                self.config.max_response_length is not None
                and agent_data.total_response_length >= self.config.max_response_length
            ):
                return AgentState.TERMINATED
            agent_data.messages.append({"role": "tool", "content": response})

        agent_data.user_turns += 1
        return AgentState.GENERATING

    def _call_tool(self, tool_call: ToolCall) -> str:
        return search(tool_call.arguments.get("query_list", []), top_k=self.config.top_k)
