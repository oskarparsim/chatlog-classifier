import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CLASSIFIED = BASE_DIR / "outputs/classified_relevant.json"
DEFAULT_CONVERSATIONS = BASE_DIR / "data/conversations.json"
DEFAULT_OUTPUT = BASE_DIR / "outputs/relevant_prompt_stats.json"


def load_json_list(path: Path) -> List[Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "conversations" in data:
        data = data["conversations"]
    if not isinstance(data, list):
        raise ValueError(f"Expected list-like JSON in {path}")
    return data


def content_length(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(content_length(item) for item in content)
    if isinstance(content, dict):
        total = 0
        parts = content.get("parts")
        if isinstance(parts, list):
            total += sum(content_length(part) for part in parts)
        if "text" in content and isinstance(content["text"], str):
            total += len(content["text"])
        if "content" in content:
            total += content_length(content["content"])
        return total
    return 0


def iter_user_messages(conversation: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    mapping = conversation.get("mapping", {})
    for node in mapping.values():
        message = node.get("message")
        if not message:
            continue
        author = (message.get("author") or {}).get("role")
        if author == "user":
            yield message


def per_conversation_stats(
    classified: List[Dict[str, Any]],
    conversations: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    per_conv: List[Dict[str, Any]] = []
    total_prompts = 0
    total_chars = 0

    for entry in classified:
        idx = entry["index"]
        try:
            conversation = conversations[idx]
        except IndexError as exc:
            raise IndexError(f"Conversation index {idx} missing in conversations file") from exc

        user_prompts = 0
        user_chars = 0
        for message in iter_user_messages(conversation):
            user_prompts += 1
            user_chars += content_length(message.get("content"))

        avg_prompt_len = user_chars / user_prompts if user_prompts else 0
        total_prompts += user_prompts
        total_chars += user_chars

        per_conv.append(
            {
                "index": idx,
                "title": entry.get("title") or f"Conversation {idx}",
                "number_of_prompts": user_prompts,
                "total_prompt_length": user_chars,
                "average_prompt_length": avg_prompt_len,
            }
        )

    conversation_count = len(per_conv)
    prompts_avg = total_prompts / conversation_count if conversation_count else 0
    prompt_length_avg = total_chars / total_prompts if total_prompts else 0

    if conversation_count:
        extreme_count = max(1, math.ceil(conversation_count * 0.01))
        sorted_convs = sorted(per_conv, key=lambda item: item["total_prompt_length"])
        shortest = sorted_convs[:extreme_count]
        longest = list(reversed(sorted_convs[-extreme_count:]))
    else:
        extreme_count = 0
        shortest = []
        longest = []

    summary = {
        "conversations_analyzed": conversation_count,
        "total_user_prompts": total_prompts,
        "total_user_prompt_characters": total_chars,
        "overall_average_number_of_prompts": prompts_avg,
        "overall_average_prompt_length": prompt_length_avg,
        "longest_conversations_top_1_percent": longest,
        "shortest_conversations_bottom_1_percent": shortest,
        "extreme_bucket_count": extreme_count,
    }

    return per_conv, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute prompt statistics for classified ChatGPT conversations."
    )
    parser.add_argument(
        "--classified",
        type=Path,
        default=DEFAULT_CLASSIFIED,
        help="Path to classified conversations JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--conversations",
        type=Path,
        default=DEFAULT_CONVERSATIONS,
        help="Path to raw conversations JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to store the computed statistics JSON (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classified = load_json_list(args.classified)
    conversations = load_json_list(args.conversations)
    per_conv, summary = per_conversation_stats(classified, conversations)
    output = {"per_conversation": per_conv, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output} for {summary['conversations_analyzed']} conversations.")


if __name__ == "__main__":
    main()
