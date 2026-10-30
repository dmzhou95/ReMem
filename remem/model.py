import torch
from torch import nn
from .backbones import make_backbone
from .router import UtilityRouter
from .memory import EpisodicMemory
from .restoration import RestorationController


class ReMem(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = make_backbone(cfg)
        dim = self.backbone.dim
        self.init = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, 2, bias=False)
        self.router = UtilityRouter(dim, cfg.kappa, cfg.temperature)
        self.memory = EpisodicMemory(dim, cfg.episode_size)
        self.restorer = RestorationController(cfg.restore_threshold)

    def forward(self, target, history, teacher=False):
        if not history:
            raise ValueError("ReMem requires at least one historical behavior")
        history = history[-self.cfg.max_history:]
        encoded = self.backbone.encode([target] + history)
        c, x = encoded[0], encoded[1:]
        n = len(x)
        h = self.init(c)
        full_h = h.detach()
        restored = torch.zeros(n, dtype=torch.bool, device=x.device)
        records = []
        for _ in range(self.cfg.steps):
            utility, hard, gate = self.router(x, c, h)
            # Restored tokens take precedence: A and R are always disjoint.
            active = hard & ~restored
            inactive = ~(active | restored)
            episodes = self.memory(x, utility, inactive, c, h)
            active_x = x[active] * gate[active, None]
            old_restored_x = x[restored]
            pre_parts = [c[None], active_x]
            pre_parts.extend(e.memory[None] for e in episodes)
            pre_parts.append(old_restored_x)
            pre_h = self.backbone.reason(torch.cat(pre_parts), h)
            pre_logits = self.head(pre_h)
            scores, reopen, restore_gate = self.restorer(pre_logits, pre_h, episodes, utility)
            new_restored = restored.clone()
            remaining, reopened = [], []
            for k, e in enumerate(episodes):
                if bool(reopen[k]):
                    new_restored[e.indices] = True
                    reopened.append((e.indices, x[e.indices] * restore_gate[k]))
                else:
                    remaining.append(e.memory[None] * (1 - restore_gate[k]))
            if bool(reopen.any()):
                # Keep restored tokens in original chronological order.
                ri = [torch.where(restored)[0]] + [idx for idx, _ in reopened]
                rv = [old_restored_x] + [v for _, v in reopened]
                indices, vectors = torch.cat(ri), torch.cat(rv)
                all_restored = vectors[torch.argsort(indices)]
                post_context = torch.cat([c[None], active_x] + remaining + [all_restored])
                post_h = self.backbone.reason(post_context, h)
            else:
                post_h = pre_h
            post_logits = self.head(post_h)
            teacher_h, teacher_logits = None, None
            if teacher:
                with torch.no_grad():
                    full_h = self.backbone.reason(torch.cat([c[None], x.detach()]), full_h)
                    teacher_h = full_h
                    teacher_logits = self.head(full_h)
            pre_length = int(active.sum() + restored.sum()) + len(episodes)
            post_length = int(active.sum() + new_restored.sum()) + len(remaining)
            # Exact hard forward length with a documented local surrogate.
            # Episode partitions themselves are not differentiated.
            budget_length = gate[~restored].sum() + restored.sum() + len(episodes)
            records.append(dict(pre_logits=pre_logits, logits=post_logits, state=post_h,
                teacher_state=teacher_h, teacher_logits=teacher_logits, episodes=episodes,
                scores=scores, reopen=reopen, restore_gate=restore_gate,
                pre_length=pre_length, post_length=post_length, budget_length=budget_length,
                active=active, restored=new_restored.clone(), utility=utility))
            h, restored = post_h, new_restored
        return {"logits": records[-1]["logits"], "steps": records, "n": n}
