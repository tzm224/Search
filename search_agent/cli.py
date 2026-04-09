from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_BASE_URL,
    DEFAULT_GRADER_MODEL_NAME,
    GenerationConfig,
    JudgingConfig,
    load_json_config,
    to_path,
)
from .workflows import generate_predictions, judge_predictions, report_from_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if not hasattr(args, "command") or args.command is None:
        parser.print_help()
        return 1

    config_data = load_json_config(getattr(args, "config", None))

    if args.command == "generate":
        config = build_generation_config(args, config_data)
        generate_predictions(config)
        return 0

    if args.command == "judge":
        config = build_judging_config(args, config_data)
        judge_predictions(config)
        return 0

    if args.command == "report":
        input_path = require_path(
            args.input_path or config_data.get("input_path"),
            "input_path",
        )
        summary = report_from_path(str(input_path))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "pipeline":
        generation_config = build_generation_config(args, config_data, pipeline_mode=True)
        judging_config = build_judging_config(args, config_data, pipeline_mode=True)
        generate_predictions(generation_config)
        judge_predictions(judging_config)
        summary = report_from_path(str(judging_config.output_path))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="search_agent",
        description="search_agent command line interface",
    )
    subparsers = parser.add_subparsers(dest="command")

    generate_parser = subparsers.add_parser("generate", help="Generate model responses")
    add_generation_args(generate_parser, include_pipeline_paths=False)

    judge_parser = subparsers.add_parser("judge", help="Judge model responses")
    add_judging_args(judge_parser, include_pipeline_paths=False)

    report_parser = subparsers.add_parser("report", help="Report accuracy summary")
    report_parser.add_argument("--config", type=Path, default=None)
    report_parser.add_argument("--input-path", type=Path, default=None)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run generate + judge + report")
    add_generation_args(pipeline_parser, include_pipeline_paths=True)
    add_judging_args(pipeline_parser, include_pipeline_paths=True)

    return parser


def add_generation_args(parser: argparse.ArgumentParser, *, include_pipeline_paths: bool) -> None:
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--input-path", type=Path, default=None)
    if include_pipeline_paths:
        parser.add_argument("--prediction-output-path", type=Path, default=None)
    else:
        parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["auto", "agent", "direct"], default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--concurrent", type=int, default=None)
    parser.add_argument("--max-response-length", type=int, default=None)
    parser.add_argument("--max-assistant-turns", type=int, default=None)
    parser.add_argument("--max-user-turns", type=int, default=None)
    parser.add_argument("--unique-key", type=str, default=None)


def add_judging_args(parser: argparse.ArgumentParser, *, include_pipeline_paths: bool) -> None:
    if include_pipeline_paths:
        parser.add_argument("--evaluation-output-path", type=Path, default=None)
        parser.add_argument("--judge-input-path", type=Path, default=None)
        parser.add_argument("--judge-concurrent", type=int, default=None)
    else:
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--input-path", type=Path, default=None)
        parser.add_argument("--output-path", type=Path, default=None)
        parser.add_argument("--concurrent", type=int, default=None)

    parser.add_argument("--grader-base-url", type=str, default=None)
    parser.add_argument("--grader-model-name", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)


def build_generation_config(
    args: argparse.Namespace,
    config_data: dict[str, Any],
    *,
    pipeline_mode: bool = False,
) -> GenerationConfig:
    input_path = require_path(
        args.input_path or config_data.get("input_path"),
        "input_path",
    )
    output_value = (
        args.prediction_output_path if pipeline_mode else args.output_path
    ) or config_data.get("prediction_output_path") or config_data.get("output_path")
    output_path = require_path(output_value, "output_path")

    model_name = require_value(args.model_name or config_data.get("model_name"), "model_name")
    return GenerationConfig(
        input_path=input_path,
        output_path=output_path,
        base_url=args.base_url or config_data.get("base_url") or DEFAULT_BASE_URL,
        model_name=model_name,
        top_k=args.top_k or config_data.get("top_k") or 5,
        concurrent=args.concurrent or config_data.get("concurrent") or 64,
        max_response_length=coalesce(
            args.max_response_length,
            config_data.get("max_response_length"),
        ),
        max_assistant_turns=args.max_assistant_turns or config_data.get("max_assistant_turns") or 10,
        max_user_turns=args.max_user_turns or config_data.get("max_user_turns") or 10,
        unique_key=args.unique_key or config_data.get("unique_key") or "id",
        mode=args.mode or config_data.get("mode") or "auto",
    )


def build_judging_config(
    args: argparse.Namespace,
    config_data: dict[str, Any],
    *,
    pipeline_mode: bool = False,
) -> JudgingConfig:
    if pipeline_mode:
        input_value = args.judge_input_path or config_data.get("judge_input_path") or config_data.get("prediction_output_path")
        concurrent_value = args.judge_concurrent or config_data.get("judge_concurrent") or config_data.get("concurrent")
        output_value = args.evaluation_output_path or config_data.get("evaluation_output_path")
    else:
        input_value = args.input_path or config_data.get("input_path")
        concurrent_value = args.concurrent or config_data.get("concurrent")
        output_value = args.output_path or config_data.get("output_path")

    input_path = require_path(input_value, "input_path")
    output_path = require_path(output_value, "output_path")

    grader_base_url = (
        args.grader_base_url
        or config_data.get("grader_base_url")
        or config_data.get("base_url")
        or DEFAULT_BASE_URL
    )
    grader_model_name = (
        args.grader_model_name
        or config_data.get("grader_model_name")
        or DEFAULT_GRADER_MODEL_NAME
    )

    return JudgingConfig(
        input_path=input_path,
        output_path=output_path,
        grader_base_url=grader_base_url,
        grader_model_name=grader_model_name,
        concurrent=concurrent_value or 64,
        temperature=args.temperature or config_data.get("temperature") or 0.1,
        max_tokens=args.max_tokens or config_data.get("max_tokens") or 128,
    )


def require_value(value: Any, name: str) -> Any:
    if value is None or value == "":
        raise ValueError(f"`{name}` is required")
    return value


def require_path(value: Any, name: str) -> Path:
    path = to_path(value)
    if path is None:
        raise ValueError(f"`{name}` is required")
    return path


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
