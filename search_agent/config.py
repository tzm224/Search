from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


DEFAULT_GRADER_MODEL_NAME = "Qwen2.5-32B-Instruct"
DEFAULT_BASE_URL = "127.0.0.1:8000"


@dataclass(slots=True)
class GenerationConfig:
    input_path: Path
    output_path: Path
    base_url: str
    model_name: str
    top_k: int = 5
    concurrent: int = 64
    max_response_length: int | None = None
    max_assistant_turns: int = 10
    max_user_turns: int = 10
    unique_key: str = "id"
    mode: str = "auto"


@dataclass(slots=True)
class JudgingConfig:
    input_path: Path
    output_path: Path
    grader_base_url: str = DEFAULT_BASE_URL
    grader_model_name: str = DEFAULT_GRADER_MODEL_NAME
    concurrent: int = 64
    temperature: float = 0.1
    max_tokens: int = 128


def load_json_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()
