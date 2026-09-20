#!/usr/bin/env python3
"""Stage 1 (optional): keyword pre-filter.

Scores each conversation in a ChatGPT export by counting programming/study
keywords (English and Estonian) and splits it into "matched" (likely academic)
and "unmatched" (chit-chat). Optional exclusion lists can force conversations
out by title or keyword. Cheap and offline; useful for shrinking the input
before the LLM classification stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


# Keywords chosen from recurring terms in the export (English + Estonian).
PROGRAMMING_KEYWORDS = {
    "programming": [
        "programming",
        "coding",
        "code",
        "bug",
        "fix",
        "refactor",
        "algorithm",
        "function",
        "class",
        "module",
        "package",
        "library",
        "framework",
        "api",
        "database",
        "sql",
        "python",
        "java",
        "javascript",
        "typescript",
        "c++",
        "c#",
        "rust",
        "kotlin",
        "idris",
        "swift",
        "html",
        "css",
        "bash",
        "shell",
        "linux",
        "docker",
        "git",
        "unit",
        "test",
        "testing",
        "ci",
        "deploy",
        "error",
        "exception",
        "stacktrace",
        "framework",
        "sdk",
        "oop",
        "json",
        "xml",
        "yaml",
        "regex",
        "kood",
        "programm",
        "script",
        "skript"
    ],
    "studies": [
        "homework",
        "assignment",
        "coursework",
        "lecture",
        "exam",
        "eksam",
        "quiz",
        "university",
        "college",
        "school",
        "syllabus",
        "semester",
        "midterm",
        "final",
        "praktikum",
        "projekt",
        "aine",
        "kursus",
        "kursuse",
        "ul",
        "yl",
        "uleseanne",
        "ülesanne",
        "ülesanded",
        "õping",
        "õppima",
        "õppetöö",
        "essee",
        "lõputöö",
        "tudeng",
        "õpetaja",
        "õppejõud",
        "praktika",
        "akadeemiline",
        "arvestus",
        "hindamine",
    ],
}

# Allow Estonian characters that appear in the export when normalizing text.
SAFE_CHARS = "a-z0-9õäöüšž"
TOKENIZER = re.compile(fr"[^{SAFE_CHARS}\\s]+")


BASE_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split conversations.json into programming/studies vs other content."
    )
    parser.add_argument(
        "--input",
        default=str(BASE_DIR / "data" / "conversations.json"),
        help="Path to the original conversations export.",
    )
    parser.add_argument(
        "--matched-output",
        default=str(BASE_DIR / "outputs" / "prefilter" / "matched.json"),
        help="Where to store conversations matched as academic (programming/studies).",
    )
    parser.add_argument(
        "--unmatched-output",
        default=str(BASE_DIR / "outputs" / "prefilter" / "unmatched.json"),
        help="Where to store everything else.",
    )
    parser.add_argument(
        "--report",
        default=str(BASE_DIR / "outputs" / "prefilter" / "report.csv"),
        help="CSV report containing every conversation and its assigned bucket.",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=3,
        help="Minimum keyword score required to classify something as programming/studies.",
    )
    parser.add_argument(
        "--exclude-titles",
        default=None,
        help="Optional path to newline-delimited titles to always treat as chit-chat.",
    )
    parser.add_argument(
        "--exclude-keywords",
        default=None,
        help="Optional path to newline-delimited keywords; if a conversation contains any, force it to chit-chat.",
    )
    return parser.parse_args()


def flatten_text(mapping: Dict) -> str:
    """Collect every textual part from the conversation tree."""
    parts: List[str] = []
    for node in mapping.values():
        message = node.get("message")
        if not message:
            continue
        content = message.get("content") or {}
        if content.get("content_type") != "text":
            continue
        for part in content.get("parts") or []:
            if isinstance(part, str):
                parts.append(part)
    return "\n".join(parts)


def normalize(text: str) -> List[str]:
    lowered = text.lower()
    cleaned = TOKENIZER.sub(" ", lowered)
    tokens = cleaned.split()
    return tokens


def score_tokens(tokens: Iterable[str]) -> Tuple[int, Dict[str, int]]:
    counts = Counter(tokens)
    hits = {
        bucket: sum(counts[word] for word in keywords)
        for bucket, keywords in PROGRAMMING_KEYWORDS.items()
    }
    total = sum(hits.values())
    return total, hits


def load_excluded_titles(path: str | None) -> Set[str]:
    if not path:
        return set()
    file_path = Path(path)
    if not file_path.exists():
        return set()
    return {
        line.strip().casefold()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def load_blacklist(path: str | None) -> Set[str]:
    if not path:
        return set()
    file_path = Path(path)
    if not file_path.exists():
        return set()
    return {
        line.strip().casefold()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def categorize(
    conversations: List[Dict],
    min_score: int,
    excluded_titles: Set[str],
    blacklist_keywords: Set[str],
) -> Tuple[List[Dict], List[Dict], List[Dict[str, str]]]:
    programming: List[Dict] = []
    other: List[Dict] = []
    report_rows: List[Dict[str, str]] = []

    for convo in conversations:
        text = flatten_text(convo.get("mapping", {}))
        tokens = normalize(text)
        score, hits = score_tokens(tokens)
        title = convo.get("title") or ""
        normalized_title = title.casefold()
        blacklist_hits = sorted({token for token in tokens if token in blacklist_keywords})
        bucket = "programming_studies" if score >= min_score else "other"
        override = ""
        if bucket == "programming_studies":
            if normalized_title in excluded_titles:
                bucket = "other"
                override = "title_exclusion"
            elif blacklist_hits:
                bucket = "other"
                override = f"keyword_blacklist:{','.join(blacklist_hits)}"

        report_rows.append(
            {
                "title": title,
                "score": str(score),
                "programming_hits": str(hits["programming"]),
                "study_hits": str(hits["studies"]),
                "bucket": bucket,
                "override": override,
            }
        )
        if bucket == "programming_studies":
            programming.append(convo)
        else:
            other.append(convo)
    return programming, other, report_rows


def write_json(path: Path, payload: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not payload:
        path.write_text("[]", encoding="utf-8")
        return
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["title", "score", "programming_hits", "study_hits", "bucket", "override"]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_path = Path(args.input)
    conversations = json.loads(source_path.read_text(encoding="utf-8"))

    excluded_titles = load_excluded_titles(args.exclude_titles)
    blacklist_keywords = load_blacklist(args.exclude_keywords)
    prog, other, report_rows = categorize(
        conversations, args.min_score, excluded_titles, blacklist_keywords
    )

    write_json(Path(args.matched_output), prog)
    write_json(Path(args.unmatched_output), other)
    write_report(Path(args.report), report_rows)

    forced_count = sum(1 for row in report_rows if row.get("override"))
    print(
        f"Labeled {len(prog)} / {len(conversations)} conversations as programming/studies "
        f"(forced other: {forced_count}). Report written to {args.report}."
    )


if __name__ == "__main__":
    main()
