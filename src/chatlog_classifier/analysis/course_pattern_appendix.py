import argparse
import csv
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = BASE_DIR / "outputs/classified_relevant_enriched.json"
DEFAULT_JSON_OUTPUT = BASE_DIR / "outputs/course_pattern_appendix.json"
DEFAULT_CSV_OUTPUT = BASE_DIR / "outputs/course_pattern_appendix.csv"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def course_label(course: dict) -> str:
    code = (course.get("course_code") or "").strip()
    name = (course.get("course_name") or "").strip()
    if code and name:
        return f"{code} | {name}"
    return code or name


def empty_row(label: str) -> dict:
    return {
        "course": label,
        "conversation_count": 0,
        "one_tag_count": 0,
        "two_tag_count": 0,
        "three_tag_count": 0,
        "exclusive_debugging_count": 0,
        "exclusive_code_generation_count": 0,
        "exclusive_explanations_count": 0,
        "debugging_count": 0,
        "code_generation_count": 0,
        "explanations_count": 0,
        "dominant_pattern": "balanced",
        "debugging_minus_code_generation": 0,
    }


def build_rows(conversations: list[dict]) -> list[dict]:
    per_course: dict[str, dict] = {}

    for conversation in conversations:
        raw_tags = {(tag or "").strip().lower() for tag in (conversation.get("tags") or [])}
        tags = {
            "debugging"
            if tag == "debugging"
            else "code_generation"
            if tag == "code generation"
            else "explanations"
            if tag == "explanations"
            else None
            for tag in raw_tags
        }
        tags.discard(None)

        matched_courses = conversation.get("matched_courses") or []
        labels = {
            course_label(course)
            for course in matched_courses
            if isinstance(course, dict)
        }
        labels = {label for label in labels if label}
        if not labels:
            continue

        for label in labels:
            row = per_course.setdefault(label, empty_row(label))
            row["conversation_count"] += 1

            tag_count = len(tags)
            if tag_count == 1:
                row["one_tag_count"] += 1
            elif tag_count == 2:
                row["two_tag_count"] += 1
            elif tag_count >= 3:
                row["three_tag_count"] += 1

            if "debugging" in tags:
                row["debugging_count"] += 1
            if "code_generation" in tags:
                row["code_generation_count"] += 1
            if "explanations" in tags:
                row["explanations_count"] += 1

            if tags == {"debugging"}:
                row["exclusive_debugging_count"] += 1
            elif tags == {"code_generation"}:
                row["exclusive_code_generation_count"] += 1
            elif tags == {"explanations"}:
                row["exclusive_explanations_count"] += 1

    rows = []
    for row in per_course.values():
        diff = row["debugging_count"] - row["code_generation_count"]
        row["debugging_minus_code_generation"] = diff
        if diff > 0:
            row["dominant_pattern"] = "debugging_heavier"
        elif diff < 0:
            row["dominant_pattern"] = "code_generation_heavier"
        else:
            row["dominant_pattern"] = "balanced"
        rows.append(row)

    rows.sort(key=lambda item: (-item["conversation_count"], item["course"]))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "course",
        "conversation_count",
        "one_tag_count",
        "two_tag_count",
        "three_tag_count",
        "exclusive_debugging_count",
        "exclusive_code_generation_count",
        "exclusive_explanations_count",
        "debugging_count",
        "code_generation_count",
        "explanations_count",
        "dominant_pattern",
        "debugging_minus_code_generation",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an appendix-ready per-course tag pattern summary."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSON path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help=f"JSON output path (default: {DEFAULT_JSON_OUTPUT})",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
        help=f"CSV output path (default: {DEFAULT_CSV_OUTPUT})",
    )
    args = parser.parse_args()

    conversations = load_json(args.input)
    if not isinstance(conversations, list):
        raise ValueError("Input JSON must be a list of conversation objects.")

    rows = build_rows(conversations)
    payload = {
        "total_courses": len(rows),
        "courses": rows,
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.csv_output, rows)

    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.csv_output}")


if __name__ == "__main__":
    main()
