import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search_agent.config import DEFAULT_BASE_URL, DEFAULT_GRADER_MODEL_NAME, JudgingConfig
from search_agent.workflows import judge_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge search_agent responses")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="test_data.jsonl",
        help="dataset filename used in eval/get_response.py",
    )
    parser.add_argument("--concurrent", type=int, default=256)
    parser.add_argument("--grader_base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--grader_model_name", type=str, default=DEFAULT_GRADER_MODEL_NAME)

    args = parser.parse_args()

    config = JudgingConfig(
        input_path=ROOT / "output" / f"{args.model_name}.{args.dataset}",
        output_path=ROOT / "eval_result" / f"{args.model_name}.{args.dataset}",
        grader_base_url=args.grader_base_url,
        grader_model_name=args.grader_model_name,
        concurrent=args.concurrent,
    )
    judge_predictions(config)

                

        
