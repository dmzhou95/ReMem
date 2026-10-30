"""Training with per-example dynamic contexts and gradient accumulation."""
import argparse
import json
import random
from pathlib import Path
import torch
from remem.config import Config
from remem.data import load_jsonl
from remem.model import ReMem
from remem.losses import compute_loss
from remem.utils import seed_everything, choose_device, save_checkpoint
from remem.metrics import evaluate_metrics


@torch.no_grad()
def validate(model, rows):
    model.eval()
    scores = [float(model(r["target"], r["history"])["logits"].softmax(-1)[1]) for r in rows]
    return evaluate_metrics([r["label"] for r in rows], scores, [r["group_id"] for r in rows])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/tiny.json")
    p.add_argument("--train", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--output", default="runs/demo")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    cfg = Config.load(args.config)
    seed_everything(cfg.seed)
    torch.set_num_threads(2)
    train, valid = load_jsonl(args.train), load_jsonl(args.valid)
    model = ReMem(cfg).to(choose_device(args.device))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")
    best = -float("inf")
    with (out / "train_log.jsonl").open("w") as log:
        for epoch in range(cfg.epochs):
            model.train()
            random.shuffle(train)
            total_loss = 0.0
            for start in range(0, len(train), cfg.batch_size):
                batch = train[start:start+cfg.batch_size]
                optimizer.zero_grad(set_to_none=True)
                for row in batch:
                    result = model(row["target"], row["history"], teacher=True)
                    loss, components = compute_loss(result, row["label"], cfg)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite training loss")
                    (loss / len(batch)).backward()
                    total_loss += float(loss.detach())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            metrics = validate(model, valid)
            score = metrics["auc"] if metrics["auc"] is not None else -metrics["logloss"]
            if score > best:
                best = score
                save_checkpoint(out / "best.pt", model, cfg, epoch)
            save_checkpoint(out / "last.pt", model, cfg, epoch)
            record = {"epoch": epoch+1, "train_loss": total_loss/len(train), "validation": metrics}
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
