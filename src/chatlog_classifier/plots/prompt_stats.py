import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CLASSIFIED = BASE_DIR / "outputs/classified_relevant.json"
DEFAULT_PROMPT_STATS = BASE_DIR / "outputs/relevant_prompt_stats.json"
DEFAULT_CONVERSATIONS = BASE_DIR / "data/conversations.json"
DEFAULT_OUT_DIR = BASE_DIR / "outputs/figures"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def month_from_conversation(conversation: dict) -> Optional[datetime]:
    timestamp = conversation.get("create_time") or conversation.get("update_time")
    if timestamp is None:
        return None
    dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def count_months(conversations: List[dict], indices: Optional[Iterable[int]] = None) -> Counter:
    counter = Counter()
    if indices is None:
        iterable = enumerate(conversations)
    else:
        iterable = ((idx, conversations[idx]) for idx in indices if 0 <= idx < len(conversations))

    for _, conversation in iterable:
        month_key = month_from_conversation(conversation)
        if month_key:
            counter[month_key] += 1
    return counter


def main():
    parser = argparse.ArgumentParser(description="Plot conversation length, use-type and monthly charts.")
    parser.add_argument("--classified", type=Path, default=DEFAULT_CLASSIFIED)
    parser.add_argument("--prompt-stats", type=Path, default=DEFAULT_PROMPT_STATS)
    parser.add_argument("--conversations", type=Path, default=DEFAULT_CONVERSATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    classified_relevant = load_json(args.classified)
    prompt_stats = load_json(args.prompt_stats)
    conversations = load_json(args.conversations)

    length_map = {
        item["index"]: item["total_prompt_length"]
        for item in prompt_stats.get("per_conversation", [])
        if "index" in item and "total_prompt_length" in item
    }

    use_type_counter = Counter()
    for entry in classified_relevant:
        for use_type in entry.get("use_types") or []:
            use_type_counter[use_type] += 1

    relevant_indices = [entry["index"] for entry in classified_relevant]
    relevant_months = count_months(conversations, relevant_indices)

    # Conversation length distribution (highly right-skewed): use log-scaled x-axis
    lengths_all = [v for v in length_map.values() if v is not None]
    lengths = [v for v in lengths_all if v > 0]

    if lengths:
        max_len = max(lengths)
        # Log-spaced bins make the long tail visible while keeping counts honest.
        bins = np.logspace(0, np.log10(max_len), 50)  # 10^0 = 1 character

        plt.figure(figsize=(10, 5))
        plt.hist(lengths, bins=bins)
        plt.xscale("log")
        plt.xlim(1, max_len)
        plt.title("Distribution of Conversation Lengths")
        plt.xlabel("Total Prompt Length (characters)")
        plt.ylabel("Number of Conversations")
        plt.tight_layout()
        plt.savefig(args.output_dir / "conversation_lengths.png", dpi=150)
        plt.close()

        # Print simple summary stats
        arr = np.array(lengths, dtype=float)
        print(
            "Conversation length stats (positive only): "
            f"n={arr.size}, min={arr.min():.0f}, p25={np.percentile(arr, 25):.0f}, "
            f"median={np.median(arr):.0f}, mean={arr.mean():.0f}, p75={np.percentile(arr, 75):.0f}, max={arr.max():.0f}"
        )
    else:
        print("No positive conversation lengths found; skipping length distribution plot.")

    plt.figure(figsize=(10, 5))
    labels = list(use_type_counter.keys())
    counts = [use_type_counter[label] for label in labels]
    plt.bar(labels, counts)
    plt.title("Use-Type Frequency")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(args.output_dir / "use_type_frequency.png", dpi=150)
    plt.close()

    month_keys = sorted(relevant_months)
    if month_keys:
        labels = [dt.strftime("%Y-%m") for dt in month_keys]
        rel_counts = [relevant_months.get(dt, 0) for dt in month_keys]

        plt.figure(figsize=(12, 5))
        plt.plot(labels, rel_counts, marker="o", color="tab:blue")
        plt.title("Monthly Relevant Conversation Counts")
        plt.xlabel("Month")
        plt.ylabel("Number of Conversations")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(args.output_dir / "monthly_conversations.png", dpi=150)
        plt.close()

    print(f"Wrote charts to {args.output_dir}")


if __name__ == "__main__":
    main()
