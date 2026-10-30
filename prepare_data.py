"""Convert normalized interactions CSV to leakage-conscious JSONL.

Required CSV columns: user_id,item_id,timestamp,text,label.
Optional impression_id groups candidates shown in the same request.
Equal timestamps share the same past history; current outcomes never enter it.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from remem.data import write_jsonl


def convert(input_path, output_path, min_interactions=10, max_history=128):
    if min_interactions < 1 or max_history < 1:
        raise ValueError("min_interactions and max_history must be positive")
    users = defaultdict(list)
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"user_id", "item_id", "timestamp", "text", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires {sorted(required)}")
        for r in reader:
            r["timestamp"] = float(r["timestamp"])
            r["label"] = int(r["label"])
            if r["label"] not in (0, 1):
                raise ValueError("label must be binary")
            users[r["user_id"]].append(r)
    splits = {name: [] for name in ("train", "valid", "test")}
    for user, rows in sorted(users.items()):
        if len(rows) < min_interactions:
            continue
        events = defaultdict(list)
        for row in rows:
            events[row["timestamp"]].append(row)
        times = sorted(events)
        history = []
        for i, timestamp in enumerate(times):
            split = "train" if i < int(.8 * len(times)) else "valid" if i < int(.9 * len(times)) else "test"
            current = events[timestamp]
            for row in current:
                if history:
                    group = row.get("impression_id") or str(timestamp)
                    splits[split].append({"user_id": user, "item_id": row["item_id"],
                        "group_id": user + ":" + group, "history": history[-max_history:].copy(),
                        "target": row["text"], "label": row["label"]})
            # Only observed positive behaviors enter history. This is an explicit convention.
            history.extend(r["text"] for r in current if r["label"] == 1)
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        write_jsonl(out / (split + ".jsonl"), records)
        print(f"{split}: {len(records)} samples")
    return splits


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="data/processed")
    p.add_argument("--min-interactions", type=int, default=10)
    p.add_argument("--max-history", type=int, default=128)
    args = p.parse_args()
    convert(args.input, args.output, args.min_interactions, args.max_history)
