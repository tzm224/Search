from __future__ import annotations

from collections import Counter
import concurrent.futures
import logging
from typing import Any, Callable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal environments
    def tqdm(iterable, **_: Any):  # type: ignore[misc]
        return iterable

from .agent import ToolAgentLoop
from .config import GenerationConfig, JudgingConfig
from .dataset import load_jsonl, write_jsonl
from .model_client import request_model
from .parser import extract_prediction_text
from .prompts import GRADER_TEMPLATE


LOGGER = logging.getLogger(__name__)


def generate_predictions(config: GenerationConfig) -> list[dict[str, Any]]:
    records = load_jsonl(config.input_path)
    agent_loop = ToolAgentLoop(config)

    def worker(record: dict[str, Any]) -> dict[str, Any]:
        result = dict(record)
        try:
            response, full_response = agent_loop.run(record)
            result["response"] = response
            if full_response and full_response != response:
                result["full_response"] = full_response
        except Exception as exc:
            result["response"] = ""
            result["error"] = str(exc)
        return result

    output = _parallel_map(
        records,
        worker,
        max_workers=config.concurrent,
        description=f"Generating responses with {config.model_name}",
    )
    write_jsonl(config.output_path, output)
    LOGGER.info("saved predictions to %s", config.output_path)
    return output


def judge_predictions(config: JudgingConfig) -> list[dict[str, Any]]:
    records = load_jsonl(config.input_path)

    def worker(record: dict[str, Any]) -> dict[str, Any]:
        result = dict(record)
        raw_prediction = record.get("response") or record.get("full_response", "")
        predicted_answer = extract_prediction_text(raw_prediction)
        prompt = GRADER_TEMPLATE.format(
            question=record["query"],
            target=record["answer"],
            predicted_answer=predicted_answer,
        )
        try:
            verdict = request_model(
                config.grader_base_url,
                config.grader_model_name,
                [{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            ).strip()
        except Exception as exc:
            verdict = ""
            result["judge_error"] = str(exc)

        result["predicted_answer"] = predicted_answer
        result["eval_result"] = verdict
        return result

    output = _parallel_map(
        records,
        worker,
        max_workers=config.concurrent,
        description=f"Judging predictions with {config.grader_model_name}",
    )
    write_jsonl(config.output_path, output)
    LOGGER.info("saved eval results to %s", config.output_path)
    return output


def summarize_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    counter = Counter(record.get("eval_result", "") for record in records)
    total = len(records)
    correct = counter.get("A", 0)
    incorrect = counter.get("B", 0)
    not_attempted = counter.get("C", 0)
    unknown = total - correct - incorrect - not_attempted

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "not_attempted": not_attempted,
        "unknown": unknown,
        "accuracy": (correct / total) if total else 0.0,
    }


def report_from_path(path: str) -> dict[str, Any]:
    records = load_jsonl(path)
    return summarize_results(records)


def _parallel_map(
    records: list[dict[str, Any]],
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    max_workers: int,
    description: str,
) -> list[dict[str, Any]]:
    if not records:
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        iterator = executor.map(worker, records)
        return list(tqdm(iterator, total=len(records), desc=description))
