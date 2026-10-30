import torch
from torch import nn
from torch.nn import functional as F
from .router import straight_through


class RestorationController(nn.Module):
    """Eqs. (22)-(26), natural-log entropy and cosine conflict."""
    def __init__(self, threshold):
        super().__init__()
        self.linear = nn.Linear(3, 1)
        self.threshold = threshold

    def forward(self, logits, state, episodes, utility):
        if not episodes:
            empty = state.new_empty(0)
            return empty, empty.bool(), empty
        probability = logits.softmax(-1)
        entropy = -(probability * logits.log_softmax(-1)).sum()
        features = []
        for e in episodes:
            conflict = 1 - F.cosine_similarity(state, e.memory, dim=0, eps=1e-8)
            features.append(torch.stack([entropy, conflict, utility[e.indices].max()]))
        scores = self.linear(torch.stack(features)).sigmoid().squeeze(-1)
        hard = scores >= self.threshold
        return scores, hard, straight_through(scores, hard)
