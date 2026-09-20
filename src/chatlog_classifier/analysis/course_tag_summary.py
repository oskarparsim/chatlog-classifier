import argparse
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = BASE_DIR / "outputs/classified_relevant_enriched.json"
DEFAULT_OUTPUT = BASE_DIR / "outputs/course_tag_usage_summary.json"


def course_label(course: dict) -> str:
    code = (course.get("course_code") or "").strip()
    name = (course.get("course_name") or "").strip()
    if code and name:
        return f"{code} | {name}"
    return code or name


def build_summary(conversations: list[dict]) -> dict:
    per_course: dict[str, dict[str, int | str]] = {}
    unmatched_conversations = 0

    for conversation in conversations:
        tags = {(tag or "").strip().lower() for tag in (conversation.get("tags") or [])}
        matched_courses = conversation.get("matched_courses") or []

        labels = {course_label(course) for course in matched_courses if isinstance(course, dict)}
        labels = {label for label in labels if label}

        if not labels:
            unmatched_conversations += 1
            continue

        for label in labels:
            if label not in per_course:
                per_course[label] = {
                    "course": label,
                    "conversation_count": 0,
                    "debugging": 0,
                    "code_generation": 0,
                    "explanations": 0,
                }

            row = per_course[label]
            row["conversation_count"] += 1
            if "debugging" in tags:
                row["debugging"] += 1
            if "code generation" in tags:
                row["code_generation"] += 1
            if "explanations" in tags:
                row["explanations"] += 1

    course_rows = sorted(
        per_course.values(),
        key=lambda row: (-row["conversation_count"], row["course"]),
    )

    return {
        "total_conversations": len(conversations),
        "unmatched_conversations": unmatched_conversations,
        "course_tag_usage_by_conversation": course_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-course conversation and tag usage summary."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSON path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        conversations = json.load(f)

    if not isinstance(conversations, list):
        raise ValueError("Input JSON must be a list of conversation objects.")

    summary = build_summary(conversations)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
