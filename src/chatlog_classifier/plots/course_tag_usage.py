"""Bar charts: conversation count by course and by tag (from course_usage.py output)."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SUMMARY = BASE_DIR / "outputs/course_usage_summary.json"
DEFAULT_OUT_DIR = BASE_DIR / "outputs/figures"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_courses(course_rows, out_path):
    labels = [row["course"].split(" | ", 1)[-1].strip() for row in course_rows]
    counts = [row["conversation_count"] for row in course_rows]
    plt.figure(figsize=(12, 6))
    plt.bar(labels, counts)
    plt.title("Conversation Count by Course")
    plt.xlabel("Course")
    plt.ylabel("Number of Conversations")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_tags(tag_rows, out_path):
    labels = [row["tag"] for row in tag_rows]
    counts = [row["conversation_count"] for row in tag_rows]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts)
    plt.title("Conversation Count by Tag")
    plt.xlabel("Tag")
    plt.ylabel("Number of Conversations")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    summary = load_json(args.input)
    course_rows = summary.get("course_counts_by_conversation", [])
    tag_rows = summary.get("tag_counts_by_conversation", [])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if course_rows:
        out = args.output_dir / "conversation_count_by_course.png"
        plot_courses(course_rows, out)
        print(f"Wrote {out}")
    if tag_rows:
        out = args.output_dir / "conversation_count_by_tag.png"
        plot_tags(tag_rows, out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
