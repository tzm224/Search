import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search_agent.config import DEFAULT_BASE_URL, GenerationConfig
from search_agent.workflows import generate_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate search_agent responses")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="test_data.jsonl",
        help="dataset filename under eval/dataset",
    )
    parser.add_argument("--max_response_length", type=int, default=None)
    parser.add_argument("--max_assistant_turns", type=int, default=10)
    parser.add_argument("--unique_key", type=str, default="id")
    parser.add_argument("--max_user_turns", type=int, default=10)
    parser.add_argument("--concurrent", type=int, default=128)

    args = parser.parse_args()

    config = GenerationConfig(
        input_path=ROOT / "eval" / "dataset" / args.dataset,
        output_path=ROOT / "output" / f"{args.model_name}.{args.dataset}",
        base_url=args.base_url,
        model_name=args.model_name,
        top_k=args.top_k,
        concurrent=args.concurrent,
        max_response_length=args.max_response_length,
        max_assistant_turns=args.max_assistant_turns,
        max_user_turns=args.max_user_turns,
        unique_key=args.unique_key,
        mode="direct" if args.model_name.strip().lower() == "qwen3-8b" else "agent",
    )
    generate_predictions(config)
