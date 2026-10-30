"""Explicit JSONL schema; chronological preprocessing is in prepare_data.py."""
import json


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("target"), str):
                raise ValueError(f"line {line_number}: target must be text")
            history = row.get("history")
            if not isinstance(history, list) or not history or not all(isinstance(x, str) for x in history):
                raise ValueError(f"line {line_number}: history must be nonempty list[str]")
            if row.get("label") not in (0, 1):
                raise ValueError(f"line {line_number}: label must be 0 or 1")
            if not isinstance(row.get("group_id"), str):
                raise ValueError(f"line {line_number}: group_id must identify a ranking candidate set")
            records.append(row)
    if not records:
        raise ValueError("empty dataset")
    return records


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
