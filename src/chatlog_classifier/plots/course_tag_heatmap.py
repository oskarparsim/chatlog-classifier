#!/usr/bin/env python3
"""Generate a normalized course-tag heatmap (SVG + CSV) from course_tag_usage_summary.json."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "outputs" / "course_tag_usage_summary.json"
DEFAULT_COURSES = ROOT / "config" / "courses.example.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "figures"

TAGS = [
    ("explanations", "Explanations"),
    ("code_generation", "Code generation"),
    ("debugging", "Debugging"),
]

def load_display_names(path: Path) -> dict[str, str]:
    """Map course name -> display name from the optional `name_en` column of the courses CSV."""
    if not path or not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {
            row["name"]: row["name_en"]
            for row in csv.DictReader(f)
            if row.get("name") and row.get("name_en")
        }


def split_course(course: str, display_names: dict[str, str]) -> tuple[str, str]:
    if " | " in course:
        code, name = course.split(" | ", 1)
    else:
        code, name = "", course
    return code, display_names.get(name, name)


def color(value: float) -> str:
    """Blue scale with enough contrast for print."""
    value = max(0.0, min(1.0, value))
    r0, g0, b0 = 237, 246, 252
    r1, g1, b1 = 26, 98, 165
    r = round(r0 + (r1 - r0) * value)
    g = round(g0 + (g1 - g0) * value)
    b = round(b0 + (b1 - b0) * value)
    return f"#{r:02x}{g:02x}{b:02x}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="course_tag_usage_summary.json")
    parser.add_argument("--courses", type=Path, default=DEFAULT_COURSES, help="courses CSV with optional name_en column")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top", type=int, default=15, help="number of courses to show")
    args = parser.parse_args()

    OUT_CSV = args.output_dir / "course_tag_heatmap.csv"
    OUT_SVG = args.output_dir / "course_tag_heatmap.svg"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    display_names = load_display_names(args.courses)

    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = data["course_tag_usage_by_conversation"][: args.top]

    normalized_rows = []
    for row in rows:
        code, english_name = split_course(row["course"], display_names)
        tag_total = sum(row[key] for key, _ in TAGS)
        normalized = {
            key: (row[key] / tag_total if tag_total else 0.0)
            for key, _ in TAGS
        }
        normalized_rows.append((code, english_name, row, normalized))

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "course_code",
            "course_name_en",
            "conversation_count",
            "explanations_share",
            "code_generation_share",
            "debugging_share",
        ])
        for code, english_name, row, normalized in normalized_rows:
            writer.writerow([
                code,
                english_name,
                row["conversation_count"],
                f"{normalized['explanations']:.4f}",
                f"{normalized['code_generation']:.4f}",
                f"{normalized['debugging']:.4f}",
            ])

    left = 390
    top = 90
    cell_w = 150
    cell_h = 34
    row_gap = 8
    width = 920
    height = top + len(normalized_rows) * (cell_h + row_gap) + 230
    max_text_x = left - 14

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}",
        ".title{font-size:22px;font-weight:700}",
        ".axis{font-size:13px;font-weight:700}",
        ".course{font-size:12px}",
        ".value{font-size:12px;font-weight:700}",
        ".note{font-size:11px;fill:#52606d}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" class="title">Relative Distribution of Conversation Tags in the Top {len(normalized_rows)} Courses</text>',
        '<text x="24" y="56" class="note">Values are normalized within each course across explanations, code generation, and debugging.</text>',
    ]

    for col, (_, label) in enumerate(TAGS):
        x = left + col * cell_w + cell_w / 2
        parts.append(f'<text x="{x:.0f}" y="{top - 16}" class="axis" text-anchor="middle">{html.escape(label)}</text>')

    for i, (code, english_name, row, normalized) in enumerate(normalized_rows):
        y = top + i * (cell_h + row_gap)
        label = f"{english_name} (n={row['conversation_count']})"
        parts.append(f'<text x="{max_text_x}" y="{y + 22}" class="course" text-anchor="end">{html.escape(label)}</text>')
        for col, (key, _) in enumerate(TAGS):
            value = normalized[key]
            x = left + col * cell_w
            fill = color(value)
            text_color = "#ffffff" if value >= 0.55 else "#1f2933"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 8}" height="{cell_h}" rx="3" fill="{fill}"/>')
            parts.append(
                f'<text x="{x + (cell_w - 8) / 2:.0f}" y="{y + 22}" class="value" '
                f'text-anchor="middle" fill="{text_color}">{value * 100:.0f}%</text>'
            )

    legend_y = height - 38
    parts.append(f'<text x="{left}" y="{legend_y}" class="note">Lower share</text>')
    for i in range(7):
        v = i / 6
        parts.append(f'<rect x="{left + 78 + i * 28}" y="{legend_y - 13}" width="28" height="12" fill="{color(v)}"/>')
    parts.append(f'<text x="{left + 292}" y="{legend_y}" class="note">Higher share</text>')
    parts.append("</svg>")

    OUT_SVG.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
