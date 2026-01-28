"""Convert test_agentflow.json -> dataset.jsonl

Usage:
    python convert_agentflow_to_jsonl.py input.json output.jsonl

Behavior:
- `query` is copied from `q`.
- `gold_actions` is copied from `agt` preserving order and action names.
  * Special-case: for `find_user_id_by_email` we only keep the `email` argument
    (the `user_id` field in that action is treated as an output and omitted).
- `gold_final` is extracted from the last hyphen-bullet line in `ogt` (skips
  the "Orders affected" bullet). Falls back to the last non-empty line.

Produces a JSONL file where each line is one JSON object.
"""
import argparse
import json
import re
from typing import Any, Dict, List


from utils.arg_normalizer import normalize_action_arguments

def extract_gold_actions(agt: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for action in agt:
        name = action.get("name")
        args = action.get("arguments", {}) or {}
        # Special-case: treat find_user_id_by_email as taking only the email
        if name == "find_user_id_by_email":
            filtered = {}
            if "email" in args:
                filtered["email"] = args["email"]
        else:
            filtered = args.copy()

        # Normalize args to match tool invoke signatures
        try:
            filtered = normalize_action_arguments(name, filtered)
        except Exception:
            pass

        out.append({"name": name, "arguments": filtered})
    return out


def extract_gold_final(ogt: List[str]) -> str:
    # ogt usually contains one formatted message. We look for hyphen bullets.
    bullets: List[str] = []
    for msg in ogt:
        for ln in msg.splitlines():
            s = ln.strip()
            # Accept lines starting with '-' or '•'
            if s.startswith("-") or s.startswith("•"):
                # remove leading '-' and any whitespace
                txt = s.lstrip("-• ")
                if txt:
                    bullets.append(txt)
    # Filter out 'Orders affected' style bullets
    bullets = [b for b in bullets if not re.match(r"(?i)^orders? affected", b)]

    if bullets:
        final = bullets[-1].rstrip(". ")
        return final

    # Fallback: pick last non-empty line of the joined ogt
    all_lines = [ln.strip() for msg in ogt for ln in msg.splitlines() if ln.strip()]
    if all_lines:
        return all_lines[-1].rstrip(". ")
    return ""


def convert(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_path, "w", encoding="utf-8") as out:
        for item in data:
            query = item.get("q") or item.get("query") or ""
            agt = item.get("agt", [])
            ogt = item.get("ogt", [])

            record = {
                "query": query,
                "gold_actions": extract_gold_actions(agt),
                "gold_final": extract_gold_final(ogt),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to task.json")
    parser.add_argument("--output", help="Path to write dataset.jsonl")
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
