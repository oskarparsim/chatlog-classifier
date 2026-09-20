import argparse
import json
from collections import defaultdict
from pathlib import Path

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = BASE_DIR / "outputs/classified_relevant_enriched.json"
DEFAULT_OUT_DIR = BASE_DIR / "outputs/figures/per_course"

TAG_LABELS = ["debugging", "Code generation", "explanations"]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_course_label(course: dict) -> str:
    code = (course.get("course_code") or "").strip()
    name = (course.get("course_name") or "").strip()
    if code and name:
        return f"{code} | {name}"
    return code or name


def main():
    parser = argparse.ArgumentParser(
        description="Save a separate tag-count bar chart for each course."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to classified_relevant_enriched.json (default: {DEFAULT_INPUT})",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(args.input)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of conversation objects.")

    course_tag_counts = defaultdict(lambda: defaultdict(int))

    for entry in data:
        tags = {(tag or "").strip().lower() for tag in (entry.get("tags") or [])}
        matched_courses = entry.get("matched_courses") or []

        course_labels = {
            format_course_label(course)
            for course in matched_courses
            if isinstance(course, dict)
        }
        course_labels = {label for label in course_labels if label}

        if not course_labels:
            continue

        for course_label in course_labels:
            if "debugging" in tags:
                course_tag_counts[course_label]["debugging"] += 1
            if "code generation" in tags:
                course_tag_counts[course_label]["Code generation"] += 1
            if "explanations" in tags:
                course_tag_counts[course_label]["explanations"] += 1

    sorted_courses = sorted(
        course_tag_counts.items(),
        key=lambda kv: (
            -(kv[1]["debugging"] + kv[1]["Code generation"] + kv[1]["explanations"]),
            kv[0],
        ),
    )

    for course_label, counts in sorted_courses:
        values = [counts.get(tag, 0) for tag in TAG_LABELS]
        plt.figure(figsize=(8, 5))
        plt.bar(TAG_LABELS, values)
        plt.title(f"Conversation Count by Tag - {course_label}")
        plt.xlabel("Tag")
        plt.ylabel("Number of Conversations")
        plt.xticks(rotation=20, ha="right")
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        slug = re.sub(r"[^a-z0-9]+", "_", course_label.lower()).strip("_")
        plt.savefig(args.output_dir / f"{slug}.png", dpi=120)
        plt.close()

    print(f"Wrote {len(sorted_courses)} charts to {args.output_dir}")


if __name__ == "__main__":
    main()
