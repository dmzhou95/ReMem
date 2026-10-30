from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class Episode:
    indices: torch.Tensor
    summary: torch.Tensor
    memory: torch.Tensor


def chronological_groups(indices, max_size):
    """Split only contiguous inactive runs; never cross an active-token gap."""
    groups, current = [], []
    for i in indices.tolist():
        if current and (i != current[-1] + 1 or len(current) >= max_size):
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)
    return groups


class EpisodicMemory(nn.Module):
    """Eqs. (14)-(18); original embeddings are kept by the caller."""
    def __init__(self, dim, episode_size):
        super().__init__()
        self.episode_size = episode_size
        self.compress = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, history, utility, inactive, target, state):
        episodes = []
        indices = torch.where(inactive)[0]
        for group in chronological_groups(indices, self.episode_size):
            idx = torch.tensor(group, device=history.device, dtype=torch.long)
            beta = utility[idx].softmax(dim=0)
            summary = (beta[:, None] * history[idx]).sum(dim=0)
            memory = self.compress(torch.cat([summary, target, state]))
            episodes.append(Episode(idx, summary, memory))
        return episodes
