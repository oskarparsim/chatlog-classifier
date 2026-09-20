"""Print the use-type distribution of the relevant conversations."""

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "outputs" / "classified_relevant.json"

USE_TYPES = [
    "solving_homework_tasks",
    "understanding_code_examples",
    "studying_for_test",
    "answering_lecture_quizzes",
    "other_academic_use",
    "non_academic_use",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="classified_relevant.json")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    counter = Counter()
    for item in data:
        for t in item.get("use_types", []):
            counter[t] += 1

    total_conversations = len(data)
    print(f"Use-type distribution from {total_conversations} conversations:\n")

    for t in USE_TYPES:
        count = counter[t]
        pct = (count / total_conversations * 100) if total_conversations > 0 else 0
        print(f"{t:30} {count:5d}   ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
