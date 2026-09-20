"""Stage 2: LLM classification.

Sends each conversation in a ChatGPT export to an OpenAI model and stores a
structured label per conversation: relevance to university studies, domain,
and use types. Output is saved incrementally, so an interrupted run resumes
where it stopped.

Requires OPENAI_API_KEY in the environment.
"""

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "conversations.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "classified_conversations.json"
DEFAULT_MODEL = "gpt-5-mini"

SYSTEM_PROMPT = """
You are a research assistant helping to categorize ChatGPT conversations for a bachelor's thesis
about using generative AI in university studies (programming and mathematics).

The input is ONE conversation object in JSON format. Its structure is similar to:
- title: string
- metadata fields (you can ignore)
- messages: an array or similar structure
  Each message usually has:
    - author: "user" | "assistant" | other
    - parts: list of parts, where each part may have "text" (actual content),
      or other fields like "asset" or "transcript" that you can mostly ignore.

Your tasks:
1. Decide whether the conversation is related to university studies.
2. Decide which domain(s) the conversation belongs to: programming, math, or other.
3. Decide which use-type categories apply (multi-label) based on how the student uses ChatGPT.

Use-type categories (multi-label):
- "solving_homework_tasks": using ChatGPT to solve homework or graded assignments.
- "understanding_code_examples": asking for explanations of code or algorithms, step-by-step explanations,
  refactoring, debugging, etc.
- "studying_for_test": using ChatGPT to prepare for exams or tests, practice questions, or revise theory.
- "answering_lecture_quizzes": using ChatGPT to answer quizzes or clicker questions that are part of lectures
  or online learning platforms.
- "other_academic_use": academic use not clearly fitting the above (e.g., thesis writing, explaining theory without
  clear exam/quiz/homework context).
- "non_academic_use": personal or hobby use, not related to university studies.

Output ONLY valid JSON with this exact schema:

{
  "relevance": "relevant_to_university_studies" | "personal_irrelevant" | "mixed",
  "domain": ["programming" | "math" | "other"],
  "use_types": [
    "solving_homework_tasks" |
    "understanding_code_examples" |
    "studying_for_test" |
    "answering_lecture_quizzes" |
    "other_academic_use" |
    "non_academic_use"
  ],
  "discard_reason": null | "short explanation if irrelevant or mostly irrelevant",
  "notes": "short free-text note (1–2 sentences)"
}

Rules:
- Always return valid JSON (no comments, no trailing commas, no extra text).
- Include "non_academic_use" in use_types if the main content is personal/hobby.
- If you are unsure, choose the most reasonable labels but mention the uncertainty briefly in "notes".
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify conversations with an OpenAI model.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="ChatGPT export conversations.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to write classified_conversations.json")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--max-conversations", type=int, default=None, help="Only process the first N conversations (for testing)")
    return parser.parse_args()


def load_conversations(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # If it's wrapped like { "conversations": [...] }, unwrap it
    if isinstance(data, dict) and "conversations" in data:
        return data["conversations"]
    # Otherwise assume it's already an array
    if isinstance(data, list):
        return data

    raise ValueError("Unexpected JSON structure in conversations.json")


def classify_conversation(client: OpenAI, model: str, conv_obj: dict) -> dict:
    """Send ONE conversation object to the model and get back classification JSON."""
    conv_json_str = json.dumps(conv_obj, ensure_ascii=False)

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Here is one conversation object in JSON format:\n\n" + conv_json_str,
            },
        ],
    )

    json_text = getattr(response, "output_text", None)

    if not json_text:
        print("[DEBUG] Raw response from API:")
        try:
            print(response.model_dump_json(indent=2))
        except Exception:
            print(response)
        raise RuntimeError("Model returned no output_text")

    return json.loads(json_text)


def main():
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)

    conversations = load_conversations(args.input)
    total = len(conversations)
    if args.max_conversations is not None:
        total = min(total, args.max_conversations)

    print(f"Loaded {len(conversations)} conversations. Processing {total} of them.")

    results = []
    # Resume support: skip indices already present in the output file
    processed_indices = set()
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as f:
            results = json.load(f)
        processed_indices = {r["index"] for r in results}
        print(f"Found existing output with {len(processed_indices)} classified conversations.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    for idx in range(total):
        if idx in processed_indices:
            continue

        conv = conversations[idx]
        title = conv.get("title", "(no title)")
        print(f"\n[{idx+1}/{total}] Classifying conversation: {title}")

        try:
            classification = classify_conversation(client, args.model, conv)
        except Exception as e:
            print(f"Error on index {idx}: {e}")
            time.sleep(5)  # back off on rate limits / network errors
            continue

        classification["index"] = idx
        classification["title"] = title
        results.append(classification)

        # Save incrementally so progress survives a crash
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(0.2)

    print(f"\nDone. Classified {len(results)} conversations. Output saved to {args.output}")


if __name__ == "__main__":
    main()
