from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 64
    dropout: float = 0.0
    version: str = "0.1.0"
    approximate_gelu: bool = False
    architecture: str = 'gpt2'
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        if not (8 <= self.vocab_size <= 16000 and 0 < self.d_model <= 512
                and 0 < self.n_heads <= self.d_model and self.d_model % self.n_heads == 0
                and 0 < self.n_layers <= 16 and 0 < self.d_ff <= 4096
                and 3 <= self.max_seq_len <= 64 and 0 <= self.dropout < 1 and self.version == "0.1.0"):
            raise ValueError("unsupported model configuration")
        if self.architecture not in ('gpt2','llama') or not math.isfinite(self.rope_theta) or self.rope_theta <= 0 or not 0 < self.norm_eps < 1:
            raise ValueError('unsupported architecture/normalization')
        if self.architecture == 'llama' and (self.d_model // self.n_heads) % 2:
            raise ValueError('RoPE head dimension must be even')


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_heads = c.n_heads
        self.approximate_gelu = c.approximate_gelu
        self.ln1 = nn.LayerNorm(c.d_model, eps=1e-5)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.attn_out = nn.Linear(c.d_model, c.d_model)
        self.ln2 = nn.LayerNorm(c.d_model, eps=1e-5)
        self.fc1 = nn.Linear(c.d_model, c.d_ff)
        self.fc2 = nn.Linear(c.d_ff, c.d_model)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = (a.reshape(b, t, self.n_heads, d // self.n_heads).transpose(1, 2) for a in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(d // self.n_heads)
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
        z = (scores.softmax(-1) @ v).transpose(1, 2).contiguous().reshape(b, t, d)
        x = x + self.dropout(self.attn_out(z))
        return x + self.dropout(self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh" if self.approximate_gelu else "none")))


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.architecture != 'gpt2':
            raise ValueError('use make_model for non-GPT2 architectures')
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_ln = nn.LayerNorm(config.d_model, eps=1e-5)
        self.dropout = nn.Dropout(config.dropout)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.bias)

    def forward(self, ids):
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.config.max_seq_len:
            raise ValueError("invalid sequence shape/length")
        x = self.dropout(self.token_emb(ids) + self.pos_emb(torch.arange(ids.shape[1], device=ids.device)))
        for block in self.blocks:
            x = block(x)
        return F.linear(self.final_ln(x), self.token_emb.weight)  # tied output weights

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def make_model(config):
    if config.architecture == 'llama':
        from .llama_model import LlamaTransformer
        return LlamaTransformer(config)
    return Transformer(config)


def loss_for(model, batch, label_smoothing=0.0, context_ids=None):
    logits = model(batch[:, :-1] if context_ids is None else context_ids)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), ignore_index=2,
                           label_smoothing=label_smoothing)


@torch.no_grad()
def greedy(model, difficulty, vocab):
    ids = [0, difficulty]
    for _ in range(model.config.max_seq_len - 2):
        logits = model(torch.tensor([ids], device=next(model.parameters()).device))[0, -1].clone()
        logits[[0, 2, 3, 4, 5, 6]] = -torch.inf
        token = int(logits.argmax())
        if token == 1:
            break
        ids.append(token)
    tokens = [vocab["tokens"][i] for i in ids[2:]]
    return dict(text="".join(t["surface"] for t in tokens), kana="".join(t["kana"] for t in tokens),
                segments=[{k: t[k] for k in ("surface", "kana", "pos")} for t in tokens], source="model",
                model_version=model.config.version, token_ids=ids[2:],
                reading_validation="structural_only; contextual_reading_not_verified")
