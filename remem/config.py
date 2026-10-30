from dataclasses import dataclass, asdict
import json


@dataclass
class Config:
    backend: str = "tiny"
    model_path: str = ""
    dim: int = 64
    vocab_size: int = 4096
    max_text_tokens: int = 48
    max_history: int = 128
    steps: int = 3
    episode_size: int = 8
    kappa: float = 0.5
    temperature: float = 0.3
    restore_threshold: float = 0.5
    budget_ratio: float = 0.3
    margin: float = 0.02
    lambda_pred: float = 1.0
    lambda_state: float = 0.5
    lambda_mem: float = 0.5
    lambda_budget: float = 0.05
    lambda_rest: float = 0.5
    lr: float = 0.001
    weight_decay: float = 0.01
    epochs: int = 3
    batch_size: int = 4
    seed: int = 42
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    use_lora: bool = True

    def __post_init__(self):
        if self.backend not in ("tiny", "hf"):
            raise ValueError("backend must be tiny or hf")
        if min(self.steps, self.episode_size, self.max_history, self.max_text_tokens,
               self.batch_size, self.epochs) < 1:
            raise ValueError("counts must be positive")
        if self.temperature <= 0 or not 0 < self.budget_ratio <= 1:
            raise ValueError("invalid temperature or budget")
        if not 0 <= self.restore_threshold <= 1:
            raise ValueError("restore_threshold must be in [0, 1]")
        if self.backend == "tiny" and (self.dim < 4 or self.dim % 4):
            raise ValueError("tiny dim must be a positive multiple of 4")

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
