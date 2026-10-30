import random
import torch


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def save_checkpoint(path, model, cfg, epoch):
    # Frozen HF backbone weights are not duplicated; local model_path is still required.
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()
             if cfg.backend == "tiny" or key in names or not key.startswith("backbone.")}
    from dataclasses import asdict
    torch.save({"config": asdict(cfg), "state": state, "epoch": epoch}, path)


def load_checkpoint(path, device):
    from .config import Config
    from .model import ReMem
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cfg = Config(**payload["config"])
    model = ReMem(cfg)
    if cfg.backend == "tiny":
        model.load_state_dict(payload["state"], strict=True)
    else:
        result = model.load_state_dict(payload["state"], strict=False)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        if result.unexpected_keys or trainable.intersection(result.missing_keys):
            raise ValueError("checkpoint missing trainable weights or contains unexpected keys")
    return model.to(device), cfg
