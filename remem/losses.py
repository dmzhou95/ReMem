import torch
from torch.nn import functional as F


def divergence(teacher_logits, student_logits):
    return F.kl_div(student_logits.log_softmax(-1), teacher_logits.detach().softmax(-1), reduction="sum")


def compute_loss(output, label, cfg):
    """Eqs. (28)-(35); self-teacher targets are stop-gradient."""
    steps, logits = output["steps"], output["logits"]
    if steps[0]["teacher_logits"] is None:
        raise ValueError("Training losses require teacher=True")
    zero = logits.sum() * 0
    terms = {k: [] for k in ("pred", "state", "mem", "budget", "rest")}
    for s in steps:
        pred = divergence(s["teacher_logits"], s["logits"])
        terms["pred"].append(pred)
        terms["state"].append(1 - F.cosine_similarity(s["state"], s["teacher_state"].detach(), dim=0))
        terms["mem"].extend(1 - F.cosine_similarity(e.memory, e.summary, dim=0) for e in s["episodes"])
        terms["budget"].append(F.relu(s["budget_length"] / (cfg.budget_ratio * output["n"]) - 1).square())
        if len(s["scores"]):
            soft_any = 1 - (1 - s["scores"]).prod()
            hard_any = s["reopen"].any().to(logits.dtype)
            any_gate = hard_any + soft_any - soft_any.detach()
            before = divergence(s["teacher_logits"], s["pre_logits"])
            terms["rest"].append(any_gate * F.relu(pred - before + cfg.margin))
        else:
            terms["rest"].append(zero)
    losses = {k: torch.stack(v).mean() if v else zero for k, v in terms.items()}
    losses["task"] = F.cross_entropy(logits[None], torch.tensor([label], device=logits.device))
    total = losses["task"]
    for k in terms:
        total = total + getattr(cfg, "lambda_" + k) * losses[k]
    return total, losses
