import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search_agent.workflows import report_from_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print search_agent accuracy")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="test_data.jsonl")

    args = parser.parse_args()

    summary = report_from_path(str(ROOT / "eval_result" / f"{args.model_name}.{args.dataset}"))
    print(
        f"Acc: {summary['correct']}/{summary['total']} = {summary['accuracy']}"
    )

        
