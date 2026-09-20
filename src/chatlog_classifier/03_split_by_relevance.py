"""Stage 3: split classified conversations into relevant and irrelevant sets.

"mixed" and "relevant_to_university_studies" go to the relevant file;
everything else (including unknown values) goes to the irrelevant file.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"

GOOD_VALUES = {"mixed", "relevant_to_university_studies"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUT_DIR / "classified_conversations.json")
    parser.add_argument("--relevant-output", type=Path, default=OUT_DIR / "classified_relevant.json")
    parser.add_argument("--irrelevant-output", type=Path, default=OUT_DIR / "classified_irrelevant.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    good = []
    bad = []
    for item in data:
        if item.get("relevance") in GOOD_VALUES:
            good.append(item)
        else:
            bad.append(item)

    print(f"Loaded total: {len(data)}")
    print(f"Relevant (mixed or relevant): {len(good)}")
    print(f"Irrelevant (or unknown): {len(bad)}")

    args.relevant_output.parent.mkdir(parents=True, exist_ok=True)
    with args.relevant_output.open("w", encoding="utf-8") as f:
        json.dump(good, f, ensure_ascii=False, indent=2)
    with args.irrelevant_output.open("w", encoding="utf-8") as f:
        json.dump(bad, f, ensure_ascii=False, indent=2)

    print(f"Saved:\n - {args.relevant_output}\n - {args.irrelevant_output}")


if __name__ == "__main__":
    main()
