import argparse
import json
import torch
from remem.utils import load_checkpoint, choose_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", default="examples/request.json")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    torch.set_num_threads(2)
    model, _ = load_checkpoint(args.checkpoint, choose_device(args.device))
    model.eval()
    with open(args.input, encoding="utf-8") as f:
        row = json.load(f)
    with torch.inference_mode():
        result = model(row["target"], row["history"])
    trace = [{"step": i+1, "active_indices": torch.where(s["active"])[0].tolist(),
              "restored_indices": torch.where(s["restored"])[0].tolist(),
              "pre_length": s["pre_length"], "post_length": s["post_length"],
              "episode_sources": [e.indices.tolist() for e in s["episodes"]],
              "restoration_scores": s["scores"].tolist()} for i, s in enumerate(result["steps"])]
    print(json.dumps({"positive_probability": float(result["logits"].softmax(-1)[1]), "trace": trace}, indent=2))


if __name__ == "__main__":
    main()
