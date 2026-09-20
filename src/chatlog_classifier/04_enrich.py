"""Stage 4: enrich relevant conversations.

For each conversation in classified_relevant.json, asks an OpenAI model for
the sentiment of the user's prompts, usage tags (code generation, debugging,
explanations) and the best-matching course. Candidate courses are limited to
those whose semester window contains the conversation date (see --courses).

Requires OPENAI_API_KEY in the environment.
"""
import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CLASSIFIED = BASE_DIR / "outputs/classified_relevant.json"
DEFAULT_CONVERSATIONS = BASE_DIR / "data/conversations.json"
DEFAULT_COURSES = BASE_DIR / "config/courses.example.csv"
DEFAULT_OUTPUT = BASE_DIR / "outputs/classified_relevant_enriched.json"
DEFAULT_MODEL = "gpt-5-mini"


SYSTEM_PROMPT = """
You are annotating ChatGPT conversations for a university-study analysis.

Input you receive:
- A conversation (user messages concatenated).
- A list of candidate courses for the semester window (code + name + semester range + description).

Tasks:
1) Sentiment of the user prompts (not the assistant):
   - one of: "positive", "neutral", "negative", "mixed"
2) Tags (multi-label, choose any that apply):
   - "Code generation"
   - "debugging"
   - "explanations"
3) Course matching:
   - Select the single best matching course from the candidate list that fits the conversation context.
   - Only return multiple courses if the user clearly mixes topics from different courses in the same chat.
   - If none match, return an empty list.

Return ONLY valid JSON with this schema:
{
  "sentiment": "positive" | "neutral" | "negative" | "mixed",
  "tags": ["Code generation" | "debugging" | "explanations"],
  "matched_courses": [
    {
      "course_code": "...",
      "course_name": "...",
      "course_description": "...",
      "semester_start": "YYYY-MM-DD",
      "semester_end": "YYYY-MM-DD"
    }
  ]
}
"""

COURSE_ONLY_PROMPT = """
You are matching ChatGPT conversations to university courses.

Input you receive:
- A conversation (user messages concatenated).
- A list of candidate courses for the semester window (code + name + semester range + description).

Task:
- Select the single best matching course from the candidate list that fits the conversation context.
- Only return multiple courses if the user clearly mixes topics from different courses in the same chat.
- If none match, return an empty list.

Return ONLY valid JSON with this schema:
{
  "matched_courses": [
    {
      "course_code": "...",
      "course_name": "...",
      "course_description": "...",
      "semester_start": "YYYY-MM-DD",
      "semester_end": "YYYY-MM-DD"
    }
  ]
}
"""


@dataclass
class CourseInstance:
    code: str
    name: str
    description: str
    semester_start: datetime
    semester_end: datetime

    def to_model_dict(self) -> Dict[str, str]:
        return {
            "course_code": self.code,
            "course_name": self.name,
            "course_description": self.description,
            "semester_start": self.semester_start.date().isoformat(),
            "semester_end": self.semester_end.date().isoformat(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich relevant conversations with sentiment, tags, and course matching."
    )
    parser.add_argument(
        "--classified",
        type=Path,
        default=DEFAULT_CLASSIFIED,
        help="Path to classified_relevant.json",
    )
    parser.add_argument(
        "--conversations",
        type=Path,
        default=DEFAULT_CONVERSATIONS,
        help="Path to conversations.json",
    )
    parser.add_argument(
        "--courses",
        type=Path,
        default=DEFAULT_COURSES,
        help="CSV with columns: code,name,semester_start,semester_end[,description][,name_en] (see config/courses.example.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for enriched JSON",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="OpenAI model name",
    )
    parser.add_argument(
        "--max-convs",
        type=int,
        default=None,
        help="Limit number of conversations for testing",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="Max characters of user text sent to the model",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Delay between API calls",
    )
    parser.add_argument(
        "--courses-only-missing",
        action="store_true",
        help="Only fill missing matched_courses in an existing output file",
    )
    parser.add_argument(
        "--fix-empty-course-names",
        action="store_true",
        help="Recompute matched_courses where any matched course has an empty course_name",
    )
    return parser.parse_args()


def load_json_list(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "conversations" in data:
        data = data["conversations"]
    if not isinstance(data, list):
        raise ValueError(f"Expected list-like JSON in {path}")
    return data


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)


def load_courses(path: Path) -> List[CourseInstance]:
    """Read the course list. Each row is one course offering (course + semester window)."""
    courses: List[CourseInstance] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = (row.get("code") or "").strip()
            if not code:
                continue
            courses.append(
                CourseInstance(
                    code=code,
                    name=(row.get("name") or "").strip(),
                    description=(row.get("description") or "").strip(),
                    semester_start=parse_date(row["semester_start"]),
                    semester_end=parse_date(row["semester_end"]),
                )
            )
    return courses


def extract_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(extract_text(part) for part in content)
    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        parts = content.get("parts")
        if isinstance(parts, list):
            return "\n".join(extract_text(part) for part in parts)
        if "content" in content:
            return extract_text(content["content"])
    return ""


def iter_user_text(conversation: dict) -> Iterable[str]:
    for node in conversation.get("mapping", {}).values():
        message = node.get("message")
        if not message:
            continue
        author = (message.get("author") or {}).get("role")
        if author != "user":
            continue
        yield extract_text(message.get("content"))


def conversation_month(conversation: dict) -> Optional[datetime]:
    timestamp = conversation.get("create_time") or conversation.get("update_time")
    if timestamp is None:
        return None
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc)


def candidate_courses(
    conversation: dict, courses: List[CourseInstance]
) -> List[CourseInstance]:
    conv_time = conversation_month(conversation)
    if conv_time is None:
        return []
    candidates = []
    for course in courses:
        if course.semester_start <= conv_time <= course.semester_end:
            candidates.append(course)
    return candidates


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2].strip()
    tail = text[-(max_chars // 2) :].strip()
    return head + "\n...\n" + tail


def classify_conversation(
    client: OpenAI,
    model: str,
    user_text: str,
    candidates: List[CourseInstance],
    prompt: str = SYSTEM_PROMPT,
) -> dict:
    candidate_dicts = [course.to_model_dict() for course in candidates]
    payload = {
        "conversation_text": user_text,
        "candidate_courses": candidate_dicts,
    }
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    json_text = getattr(response, "output_text", None)
    if not json_text:
        raise RuntimeError("Model returned no output_text")
    return json.loads(json_text)


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    classified = load_json_list(args.classified)
    conversations = load_json_list(args.conversations)
    courses = load_courses(args.courses)

    total = len(classified)
    if args.max_convs is not None:
        total = min(total, args.max_convs)

    existing = []
    processed_indices = set()
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        processed_indices = {item["index"] for item in existing if "index" in item}

    client = OpenAI(api_key=api_key)
    results = list(existing)

    if (args.courses_only_missing or args.fix_empty_course_names) and not existing:
        raise SystemExit("courses-only-missing/fix-empty-course-names require an existing output file.")

    if args.courses_only_missing:
        target_indices = [
            item["index"]
            for item in results
            if "index" in item and not item.get("matched_courses")
        ]
    elif args.fix_empty_course_names:
        target_indices = []
        for item in results:
            if "index" not in item:
                continue
            matched = item.get("matched_courses") or []
            if any(not course.get("course_name") for course in matched):
                target_indices.append(item["index"])
    else:
        target_indices = [classified[i]["index"] for i in range(total)]

    if args.max_convs is not None:
        target_indices = target_indices[: args.max_convs]

    for i, idx in enumerate(target_indices, start=1):
        entry = next((item for item in classified if item["index"] == idx), None)
        if entry is None:
            continue
        if not (args.courses_only_missing or args.fix_empty_course_names) and idx in processed_indices:
            continue
        conversation = conversations[idx]
        user_text = "\n".join(iter_user_text(conversation)).strip()
        if not user_text:
            user_text = "(no user text)"
        user_text = truncate_text(user_text, args.max_chars)
        candidates = candidate_courses(conversation, courses)

        print(f"[{i}/{len(target_indices)}] Enriching conversation index {idx}...")
        try:
            if args.courses_only_missing or args.fix_empty_course_names:
                prompt = COURSE_ONLY_PROMPT
            else:
                prompt = SYSTEM_PROMPT
            enriched = classify_conversation(
                client, args.model, user_text, candidates, prompt
            )
        except Exception as exc:
            print(f"Error on index {idx}: {exc}")
            time.sleep(5)
            continue

        if args.courses_only_missing or args.fix_empty_course_names:
            existing_entry = next(
                (item for item in results if item.get("index") == idx), None
            )
            if existing_entry is None:
                continue
            existing_entry["matched_courses"] = enriched.get("matched_courses", [])
        else:
            merged = dict(entry)
            merged["sentiment"] = enriched.get("sentiment")
            merged["tags"] = enriched.get("tags", [])
            merged["matched_courses"] = enriched.get("matched_courses", [])
            results.append(merged)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(args.sleep)

    print(f"Done. Enriched {len(results)} conversations. Output: {args.output}")


if __name__ == "__main__":
    main()
