from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, list[str]]


class GoogleSearchParser:
    tool_call_start_token = "<google_search>"
    tool_call_end_token = "</google_search>"

    def parse_tool_call(self, text: str) -> ToolCall | None:
        if not text or not isinstance(text, str):
            return None

        match = re.fullmatch(
            r"<google_search>\s*(.*?)\s*</google_search>",
            text.strip(),
            flags=re.DOTALL,
        )
        if match is None:
            return None

        return ToolCall(
            name="google_search",
            arguments={"query_list": [match.group(1)]},
        )

    def extract_tool_calls(self, text: str) -> tuple[str, list[ToolCall]]:
        if (
            self.tool_call_start_token not in text
            or self.tool_call_end_token not in text
        ):
            return text, []

        first_match = re.search(r"<google_search>", text)
        if first_match is None:
            return text, []

        split_index = first_match.start()
        content = text[:split_index].strip()
        remaining = text[split_index:]
        matches = re.findall(
            r"<google_search>\s*.*?\s*</google_search>",
            remaining,
            flags=re.DOTALL,
        )

        tool_calls: list[ToolCall] = []
        for item in matches:
            parsed = self.parse_tool_call(item)
            if parsed is not None:
                tool_calls.append(parsed)

        return content, tool_calls


def truncate_at_tool_call(text: str) -> str:
    if not text:
        return text

    tag = "</google_search>"
    index = text.find(tag)
    if index == -1:
        return text
    return text[: index + len(tag)]


def extract_answer_block(text: str) -> str:
    if not text:
        return ""

    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if match is None:
        return text.strip()
    return match.group(1).strip()


def strip_citations(text: str) -> str:
    if not text:
        return ""

    return re.sub(r"<cite[^>]*>.*?</cite>", "", text, flags=re.DOTALL).strip()


def extract_prediction_text(text: str) -> str:
    extracted = extract_answer_block(text)
    extracted = strip_citations(extracted)
    return re.sub(r"\s+", " ", extracted).strip()
