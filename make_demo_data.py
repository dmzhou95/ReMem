"""Synthetic smoke-test data only. Never evidence for paper benchmarks."""
import argparse
import random
from pathlib import Path
from remem.data import write_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/demo")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    topics = ["science", "history", "music", "sports"]
    for split, count in [("train", 8), ("valid", 3), ("test", 3)]:
        rows = []
        for g in range(count):
            preference = rng.choice(topics)
            history = [f"Read {preference if rng.random() < .7 else rng.choice(topics)} article {i}" for i in range(24)]
            for candidate in topics:
                rows.append({"group_id": f"synthetic-{split}-{g}", "history": history,
                             "target": f"Read {candidate} article", "label": int(candidate == preference)})
        write_jsonl(out / f"{split}.jsonl", rows)
    print(f"Synthetic data written to {out}")


if __name__ == "__main__":
    main()
