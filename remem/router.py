import torch
from torch import nn


def straight_through(probability, hard):
    """Hard forward value, local soft gradient."""
    return hard.to(probability.dtype) + probability - probability.detach()


class UtilityRouter(nn.Module):
    """Eqs. (7)-(13): population std, adaptive threshold, hard routing."""
    def __init__(self, dim, kappa, temperature):
        super().__init__()
        self.x = nn.Linear(dim, dim, bias=False)
        self.c = nn.Linear(dim, dim, bias=False)
        self.h = nn.Linear(dim, dim, bias=False)
        self.score = nn.Linear(dim, 1, bias=False)
        self.kappa, self.temperature = kappa, temperature

    def forward(self, history, target, state):
        utility = self.score(torch.tanh(self.x(history) + self.c(target) + self.h(state))).sigmoid().squeeze(-1)
        threshold = utility.mean() + self.kappa * utility.std(unbiased=False)
        soft = torch.sigmoid((utility - threshold) / self.temperature)
        hard = soft >= 0.5
        return utility, hard, straight_through(soft, hard)
