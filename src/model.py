# src/model.py
import math
import torch
import torch.nn as nn
from src.config import ModelConfig

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads

        self.W_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.register_buffer("causal_mask", torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx)))

    def forward(self, x: torch.Tensor, hook_fn=None, layer_idx: int = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_scores = attn_scores.masked_fill(self.causal_mask[:T, :T] == 0, float("-inf"))
        attn_pattern = torch.softmax(attn_scores, dim=-1)

        head_out = attn_pattern @ v  # (B, n_heads, T, d_head)

        # Intercept point for activation patching or caching
        if hook_fn is not None and layer_idx is not None:
            head_out = hook_fn(head_out, layer_idx)

        out = head_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp),
            nn.GELU(),
            nn.Linear(cfg.d_mlp, cfg.d_model)
        )

    def forward(self, x: torch.Tensor, hook_fn=None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), hook_fn=hook_fn, layer_idx=self.layer_idx)
        x = x + self.mlp(self.ln2(x))
        return x

class TinyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, hook_fn=None) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.tok_embed(input_ids) + self.pos_embed(pos)

        for block in self.blocks:
            x = block(x, hook_fn=hook_fn)

        x = self.ln_final(x)
        logits = self.unembed(x)
        return logits
