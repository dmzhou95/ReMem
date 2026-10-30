import argparse
import json
import time
from pathlib import Path
import torch
from remem.data import load_jsonl
from remem.metrics import evaluate_metrics
from remem.utils import choose_device, load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="runs/evaluation.json")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    torch.set_num_threads(2)
    device = choose_device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)
    model.eval()
    rows = load_jsonl(args.data)
    scores, lengths, latencies = [], [], []
    with torch.inference_mode():
        for _ in range(2):
            model(rows[0]["target"], rows[0]["history"])
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        for row in rows:
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = model(row["target"], row["history"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
            scores.append(float(result["logits"].softmax(-1)[1]))
            lengths.append({"n": result["n"],
                "pre": [s["pre_length"] for s in result["steps"]],
                "post": [s["post_length"] for s in result["steps"]]})
    metrics = evaluate_metrics([r["label"] for r in rows], scores, [r["group_id"] for r in rows])
    metrics.update(mean_latency_ms=sum(latencies)/len(latencies),
        peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else None,
        mean_pre_context=sum(sum(x["pre"])/len(x["pre"]) for x in lengths)/len(lengths),
        mean_post_context=sum(sum(x["post"])/len(x["post"]) for x in lengths)/len(lengths))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metrics": metrics, "scores": scores, "context_lengths": lengths,
        "latency_scope": "encoding + all routing/reasoning/restoration steps; no teacher or data loading"}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
