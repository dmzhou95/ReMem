"""A CPU demo backend and an optional local Hugging Face / PEFT backend.

Both implement encode(texts)->[N,D], reason(context,state)->[D].
The recurrent state is appended as a final readout token (engineering choice).
"""
import hashlib
import math
import re
import torch
from torch import nn


class TinyBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim, self.cfg = cfg.dim, cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        layer = nn.TransformerEncoderLayer(cfg.dim, 4, 2 * cfg.dim, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.dim)

    def encode(self, texts):
        vectors = []
        for text in texts:
            tokens = re.findall(r"\w+|[^\w\s]", text.lower())[:self.cfg.max_text_tokens] or ["<empty>"]
            ids = [int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "big") % self.cfg.vocab_size for t in tokens]
            ids = torch.tensor(ids, device=self.embedding.weight.device)
            vectors.append(self.embedding(ids).mean(0))
        return torch.stack(vectors)

    def reason(self, context, state):
        x = torch.cat([context, state[None]], dim=0)
        pos = torch.arange(len(x), device=x.device, dtype=x.dtype)[:, None]
        freq = torch.exp(torch.arange(0, self.dim, 2, device=x.device, dtype=x.dtype) * (-math.log(10000.0) / self.dim))
        pe = torch.zeros_like(x)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
        mask = torch.triu(torch.ones(len(x), len(x), device=x.device, dtype=torch.bool), diagonal=1)
        return self.norm(self.transformer((x + pe)[None], mask=mask)[0, -1])


class HFBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if not cfg.model_path:
            raise ValueError("Set model_path to an authorized local model directory")
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        model = AutoModel.from_pretrained(cfg.model_path, local_files_only=True,
                                         torch_dtype=dtype, attn_implementation="sdpa")
        if cfg.use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            model = get_peft_model(model, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        self.model, self.dim, self.cfg = model, model.config.hidden_size, cfg

    def encode(self, texts):
        # Mean input-token embeddings; one behavior = one routing unit.
        # No separate contextual encoding pass over the full history.
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=self.cfg.max_text_tokens, return_tensors="pt")
        device = self.model.get_input_embeddings().weight.device
        ids, mask = inputs["input_ids"].to(device), inputs["attention_mask"].to(device)
        embeddings = self.model.get_input_embeddings()(ids)
        return ((embeddings * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)).float()

    def reason(self, context, state):
        x = torch.cat([context, state[None]], dim=0)[None]
        x = x.to(self.model.get_input_embeddings().weight.dtype)
        out = self.model(inputs_embeds=x, attention_mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.long),
                         use_cache=False, return_dict=True)
        return out.last_hidden_state[0, -1].float()


def make_backbone(cfg):
    return TinyBackbone(cfg) if cfg.backend == "tiny" else HFBackbone(cfg)
