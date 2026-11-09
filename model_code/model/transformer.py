# Small transformer that imitates the ant brain.
#
# Tokens:  [up, down, left, right, self]
#   direction token d : Linear(5 -> d_model) of [see_food, see_home, smell_food, smell_home, last_action] in d
#                       + learned direction embedding (if cfg.dir_embedding)
#   self token        : Linear(3 -> d_model) of [on_food, at_home, carrying]
# Block (x n_layers): attention (no causal mask) then MLP, both residual.
# Readout from the final residual stream:
#   mode_logits (3), dir_logits (4), pick_logit, drop_logit
#
# forward(..., return_cache=True) also returns every intermediate, for reverse engineering.

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig

SELF = -1          # index of the self token in the token sequence


class Attention(nn.Module):
    """Plain multi-head attention, written out so the pattern can be cached."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h, self.dh = cfg.n_heads, cfg.d_head
        d = cfg.d_model
        self.W_Q = nn.Linear(d, d)
        self.W_K = nn.Linear(d, d)
        self.W_V = nn.Linear(d, d)
        self.W_O = nn.Linear(d, d)

    def forward(self, x, cache=None, prefix=""):
        B, T, D = x.shape
        split = lambda t: t.view(B, T, self.h, self.dh).transpose(1, 2)   # (B, H, T, dh)
        q, k, v = split(self.W_Q(x)), split(self.W_K(x)), split(self.W_V(x))
        scores = q @ k.transpose(-1, -2) / self.dh ** 0.5                   # (B, H, T, T)
        pattern = scores.softmax(-1)
        z = pattern @ v                                                     # (B, H, T, dh)
        out = self.W_O(z.transpose(1, 2).reshape(B, T, D))
        if cache is not None:
            cache[prefix + "attn_scores"] = scores
            cache[prefix + "attn_pattern"] = pattern                        # [b, head, query, key]
            cache[prefix + "attn_z"] = z
            cache[prefix + "attn_out"] = out
        return out


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc_in = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.fc_out = nn.Linear(cfg.d_mlp, cfg.d_model)
        self.act = nn.ReLU() if cfg.act == "relu" else nn.GELU()

    def forward(self, x, cache=None, prefix=""):
        pre = self.fc_in(x)
        post = self.act(pre)
        out = self.fc_out(post)
        if cache is not None:
            cache[prefix + "mlp_pre"] = pre
            cache[prefix + "mlp_post"] = post                               # neuron activations
            cache[prefix + "mlp_out"] = out
        return out


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.ln1 = nn.LayerNorm(cfg.d_model) if cfg.use_layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(cfg.d_model) if cfg.use_layernorm else nn.Identity()

    def forward(self, x, cache=None, prefix=""):
        x = x + self.attn(self.ln1(x), cache, prefix)
        if cache is not None:
            cache[prefix + "resid_mid"] = x
        x = x + self.mlp(self.ln2(x), cache, prefix)
        if cache is not None:
            cache[prefix + "resid_post"] = x
        return x


class BrainTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        d = cfg.d_model
        self.embed_dir = nn.Linear(cfg.n_dir_features, d)
        self.embed_self = nn.Linear(cfg.n_self_features, d)
        self.dir_pos = nn.Parameter(torch.randn(cfg.n_dir, d) * 0.02) if cfg.dir_embedding else None
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(d) if cfg.use_layernorm else nn.Identity()

        self.head_mode = nn.Linear(d, cfg.n_modes)
        self.head_dir = nn.Linear(d, cfg.n_dir) if cfg.dir_readout == "self" else nn.Linear(d, 1)
        self.head_pick = nn.Linear(d, 1)
        self.head_drop = nn.Linear(d, 1)

    def embed(self, dir_tok, self_tok):
        e_dir = self.embed_dir(dir_tok)                                     # (B, 4, d)
        if self.dir_pos is not None:
            e_dir = e_dir + self.dir_pos
        e_self = self.embed_self(self_tok).unsqueeze(1)                     # (B, 1, d)
        return torch.cat([e_dir, e_self], dim=1)                            # (B, 5, d)

    def forward(self, dir_tok, self_tok, return_cache=False):
        """dir_tok (B,4,5), self_tok (B,3) -> dict of logits [, cache]"""
        cache = {} if return_cache else None
        x = self.embed(dir_tok, self_tok)
        if cache is not None:
            cache["embed"] = x
        for i, block in enumerate(self.blocks):
            x = block(x, cache, prefix=f"blocks.{i}.")
        x = self.ln_final(x)
        if cache is not None:
            cache["final"] = x

        s = x[:, SELF]                                                      # self token, (B, d)
        if self.cfg.dir_readout == "self":
            dir_logits = self.head_dir(s)
        else:
            dir_logits = self.head_dir(x[:, : self.cfg.n_dir]).squeeze(-1)  # one score per direction token

        out = dict(
            mode_logits=self.head_mode(s),                                  # (B, 3)
            dir_logits=dir_logits,                                          # (B, 4)
            pick_logit=self.head_pick(s).squeeze(-1),                       # (B,)
            drop_logit=self.head_drop(s).squeeze(-1),                       # (B,)
        )
        return (out, cache) if return_cache else out
