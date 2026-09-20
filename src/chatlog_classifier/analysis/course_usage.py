"""Count conversations per course, tag and sentiment from the enriched data."""

import argparse
import json
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = BASE_DIR / "outputs/classified_relevant_enriched.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="classified_relevant_enriched.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    COURSE_CSV = args.output_dir / "course_usage_by_conversation.csv"
    TAG_CSV = args.output_dir / "tag_usage_by_conversation.csv"
    SENTIMENT_CSV = args.output_dir / "sentiment_usage_by_conversation.csv"
    SUMMARY_JSON = args.output_dir / "course_usage_summary.json"

    data = load_json(args.input)

    course_counter = Counter()
    tag_counter = Counter()
    sentiment_counter = Counter()
    unmatched_count = 0

    for entry in data:
        matched_courses = entry.get("matched_courses") or []
        tags = entry.get("tags") or []
        sentiment = (entry.get("sentiment") or "").strip().lower()

        if not matched_courses:
            unmatched_count += 1
        else:
            seen_courses = set()
            for course in matched_courses:
                code = course.get("course_code", "").strip()
                name = course.get("course_name", "").strip()
                key = f"{code} | {name}" if name else code
                if key:
                    seen_courses.add(key)
            for key in seen_courses:
                course_counter[key] += 1

        for tag in set(tags):
            tag_counter[tag] += 1
        if sentiment:
            sentiment_counter[sentiment] += 1

    total_conversations = len(data)

    course_rows = [
        {"course": course, "conversation_count": count}
        for course, count in course_counter.most_common()
    ]
    tag_rows = [
        {"tag": tag, "conversation_count": count}
        for tag, count in tag_counter.most_common()
    ]
    sentiment_rows = [
        {"sentiment": sentiment, "conversation_count": count}
        for sentiment, count in sentiment_counter.most_common()
    ]

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "total_conversations": total_conversations,
                "unmatched_conversations": unmatched_count,
                "course_counts_by_conversation": course_rows,
                "tag_counts_by_conversation": tag_rows,
                "sentiment_counts_by_conversation": sentiment_rows,
                "sentiment_definitions": {
                    "positive": "User expresses satisfaction, optimism, gratitude, or positive feelings about tasks or outcomes.",
                    "neutral": "User expresses little to no emotion; requests or statements are informational or procedural.",
                    "negative": "User expresses frustration, dissatisfaction, confusion, or negative feelings about tasks or outcomes.",
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with COURSE_CSV.open("w", encoding="utf-8") as f:
        f.write("course,conversation_count\n")
        for row in course_rows:
            f.write(f"\"{row['course']}\",{row['conversation_count']}\n")

    with TAG_CSV.open("w", encoding="utf-8") as f:
        f.write("tag,conversation_count\n")
        for row in tag_rows:
            f.write(f"\"{row['tag']}\",{row['conversation_count']}\n")

    with SENTIMENT_CSV.open("w", encoding="utf-8") as f:
        f.write("sentiment,conversation_count\n")
        for row in sentiment_rows:
            f.write(f"\"{row['sentiment']}\",{row['conversation_count']}\n")

    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {COURSE_CSV}")
    print(f"Wrote {TAG_CSV}")
    print(f"Wrote {SENTIMENT_CSV}")


if __name__ == "__main__":
    main()
