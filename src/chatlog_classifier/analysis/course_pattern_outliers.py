import argparse
import csv
import json
import math
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = BASE_DIR / "outputs/course_pattern_appendix.json"
DEFAULT_JSON_OUTPUT = BASE_DIR / "outputs/course_pattern_outliers.json"
DEFAULT_CSV_OUTPUT = BASE_DIR / "outputs/course_pattern_outliers.csv"


METRICS = [
    # "Exclusive ... share" means the proportion of a course's conversations
    # that have exactly one tag, namely the tag named here.
    "exclusive_debugging_share",
    "exclusive_code_generation_share",
    "exclusive_explanations_share",
    # "... share" without "exclusive" means the proportion of a course's
    # conversations that include that tag, whether alone or combined.
    "debugging_share",
    "code_generation_share",
    "explanations_share",
    # Multi-tag shares show how often a course's conversations received
    # exactly two tags or all three tags.
    "two_tag_share",
    "three_tag_share",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_share(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def stddev(values: list[float], avg: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def enrich_courses(courses: list[dict]) -> list[dict]:
    enriched = []
    for course in courses:
        total = int(course.get("conversation_count", 0) or 0)
        row = dict(course)
        row["exclusive_debugging_share"] = safe_share(
            int(course.get("exclusive_debugging_count", 0) or 0), total
        )
        row["exclusive_code_generation_share"] = safe_share(
            int(course.get("exclusive_code_generation_count", 0) or 0), total
        )
        row["exclusive_explanations_share"] = safe_share(
            int(course.get("exclusive_explanations_count", 0) or 0), total
        )
        row["debugging_share"] = safe_share(int(course.get("debugging_count", 0) or 0), total)
        row["code_generation_share"] = safe_share(
            int(course.get("code_generation_count", 0) or 0), total
        )
        row["explanations_share"] = safe_share(
            int(course.get("explanations_count", 0) or 0), total
        )
        row["two_tag_share"] = safe_share(int(course.get("two_tag_count", 0) or 0), total)
        row["three_tag_share"] = safe_share(int(course.get("three_tag_count", 0) or 0), total)
        enriched.append(row)
    return enriched


def summarize_metrics(courses: list[dict]) -> dict[str, dict[str, float]]:
    summary = {}
    for metric in METRICS:
        values = [float(course.get(metric, 0.0) or 0.0) for course in courses]
        # "mean" is the average course-level value for this metric.
        avg = mean(values)
        # "stddev" shows how spread out courses are around that mean.
        sd = stddev(values, avg)
        summary[metric] = {
            "mean": avg,
            "stddev": sd,
        }
    return summary


def z_score(value: float, avg: float, sd: float) -> float:
    if sd == 0:
        return 0.0
    # "z-score" measures how many standard deviations a course is above
    # or below the average course for one specific metric.
    return (value - avg) / sd


def build_outlier_rows(courses: list[dict], metric_summary: dict[str, dict[str, float]]) -> list[dict]:
    rows = []
    for course in courses:
        row = dict(course)
        score_parts = []
        for metric in METRICS:
            avg = metric_summary[metric]["mean"]
            sd = metric_summary[metric]["stddev"]
            score = z_score(float(course.get(metric, 0.0) or 0.0), avg, sd)
            row[f"{metric}_zscore"] = score
            score_parts.append(abs(score))

        # "overall_pattern_distance" is a simple ranking score:
        # the sum of absolute z-scores across all tracked metrics.
        row["overall_pattern_distance"] = sum(score_parts)
        # "most_unusual_metric" points to the single metric where the course
        # differs most strongly from the average course.
        row["most_unusual_metric"] = max(
            METRICS,
            key=lambda metric: abs(row[f"{metric}_zscore"]),
        )
        rows.append(row)

    rows.sort(
        key=lambda row: (-row["overall_pattern_distance"], -row["conversation_count"], row["course"])
    )
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
        "exclusive_debugging_share",
        "exclusive_code_generation_share",
        "exclusive_explanations_share",
        "debugging_share",
        "code_generation_share",
        "explanations_share",
        "two_tag_share",
        "three_tag_share",
        "overall_pattern_distance",
        "most_unusual_metric",
    ] + [f"{metric}_zscore" for metric in METRICS]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identify courses with unusually different tag patterns."
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

    payload = load_json(args.input)
    courses = payload.get("courses", [])
    if not isinstance(courses, list):
        raise ValueError("Input JSON must contain a 'courses' list.")

    enriched_courses = enrich_courses(courses)
    metric_summary = summarize_metrics(enriched_courses)
    outlier_rows = build_outlier_rows(enriched_courses, metric_summary)

    output = {
        "total_courses": len(outlier_rows),
        "metric_summary": metric_summary,
        "most_unusual_courses": outlier_rows,
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.csv_output, outlier_rows)

    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.csv_output}")


if __name__ == "__main__":
    main()
